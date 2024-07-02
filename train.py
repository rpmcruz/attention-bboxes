import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('dataset', choices=['Birds', 'StanfordCars', 'StanfordDogs'])
parser.add_argument('model', choices=['ProtoPNet', 'ViT', 'OnlyClass', 'Heatmap', 'SimpleDet', 'FasterRCNN', 'FCOS', 'DETR'])
parser.add_argument('--heatmap', choices=['GaussHeatmap', 'LogisticHeatmap'], default='GaussHeatmap')
parser.add_argument('--sigmoid', action='store_true')
parser.add_argument('--penalty-l1', type=float, default=0)
parser.add_argument('--nstdev', type=float, default=1)
parser.add_argument('--occlusion', default='encoder', choices=['none', 'encoder', 'image'])
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--fast', action='store_true')
args = parser.parse_args()

import torch
from torchvision.transforms import v2
from time import time
from tqdm import tqdm
import data, models, utils
import baseline_protopnet, baseline_vit
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA #############################

class TransformProb(torch.nn.Module):
    def __init__(self, transform, prob):
        super().__init__()
        self.transform = transform
        self.prob = prob
    def forward(self, *x):
        if torch.rand(()) < self.prob:
            x = self.transform(*x)
        return x

train_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    TransformProb(v2.RandomRotation(15), 0.33),
    TransformProb(v2.RandomAffine(0, shear=10), 0.33),
    v2.RandomPerspective(0.2, 0.33),
    v2.ColorJitter(0.2, 0.2),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
ds = getattr(data, args.dataset)('/data/toys', 'train', train_transforms)
ds_noaug = getattr(data, args.dataset)('/data/toys', 'train', test_transforms)
if args.fast:
    ds = data.SubsetLabels(ds, [0, 1])
    ds_noaug = data.SubsetLabels(ds_noaug, [0, 1])
tr = torch.utils.data.DataLoader(ds, args.batchsize, True, num_workers=4, pin_memory=True)

############################# MODEL #############################

if args.model == 'ProtoPNet':
    backbone = models.Backbone()
    model = baseline_protopnet.ProtoPNet(backbone, ds.num_classes)
    slow_opt = torch.optim.Adam(backbone.parameters(), args.lr/10)
    fast_opt = torch.optim.Adam(list(model.features.parameters()) + list(model.prototype_layer.parameters()) + list(model.fc_layer.parameters()), args.lr)
elif args.model == 'ViT':
    model = baseline_vit.ViT(ds.num_classes)
    slow_opt = torch.optim.Adam([], args.lr/10)
    fast_opt = torch.optim.Adam(model.parameters(), args.lr)
else:
    backbone = models.Backbone()
    classifier = models.Classifier(ds.num_classes)
    if args.model == 'OnlyClass':
        detection = None
        heatmap = None
    elif args.model == 'Heatmap':
        detection = models.Heatmap(args.sigmoid)
        heatmap = None
    else:
        detection = getattr(models, args.model)()
        heatmap = getattr(models, args.heatmap)(args.sigmoid)
    occlusion = 'none' if args.model == 'OnlyClass' else args.occlusion
    model = models.Occlusion(backbone, classifier, detection, heatmap, occlusion, args.adversarial)
    slow_opt = torch.optim.Adam(list(backbone.parameters()) + (list(detection.parameters()) if detection != None else []), args.lr/10)
    fast_opt = torch.optim.Adam(classifier.parameters(), args.lr)
model.to(device)

############################# LOOP #############################

visualize_batch = next(iter(torch.utils.data.DataLoader(ds_noaug, 4)))

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_losses = {}
    avg_metrics = {}
    for x, masks, y in tqdm(tr, 'stage1'):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred['class'], y)
        if args.model == 'ProtoPNet':
            stage1_loss = baseline_protopnet.stage1_loss(model, pred['features'], y)
            loss += stage1_loss
            avg_losses['stage1'] = avg_losses.get('stage1', 0) + float(stage1_loss)/len(tr)
        if 'heatmap' in pred:
            l1 = torch.mean(pred['heatmap'])
            loss += args.penalty_l1 * l1
            avg_losses['l1'] = avg_losses.get('l1', 0) + float(l1)/len(tr)
        if 'bboxes' in pred:
            avg_metrics['bbox_size'] = avg_metrics.get('bbox_size', 0) + float(torch.mean(pred['bboxes'][:, 2:]))/len(tr)
        fast_opt.zero_grad()
        slow_opt.zero_grad()
        loss.backward(retain_graph=args.adversarial)
        fast_opt.step()
        slow_opt.step()
        avg_losses['loss'] = avg_losses.get('loss', 0) + float(loss)/len(tr)
        if args.adversarial:
            # temporarily disable gradients for backbone and classifier
            for module in [backbone, classifier]:
                for param in module.parameters():
                    param.requires_grad = False
            adv_loss = torch.nn.functional.cross_entropy(pred['min_class'], y) + \
                -torch.nn.functional.cross_entropy(pred['max_class'], y)
            adv_loss.backward()
            for module in [backbone, classifier]:
                for param in module.parameters():
                    param.requires_grad = True
            avg_losses['adv'] = avg_losses.get('adv', 0) + float(adv_loss)/len(tr)
        avg_losses['loss'] = avg_losses.get('loss', 0) + float(loss)/len(tr)
        avg_metrics['acc'] = avg_metrics.get('acc', 0) + (y == pred['class'].argmax(1)).float().mean()/len(tr)
        if 'heatmap' in pred:
            masks = masks.to(device)
            heatmaps = pred['heatmap'][:, None]
            heatmaps = torch.nn.functional.interpolate(heatmaps, masks.shape[-2:], mode='nearest-exact')
            avg_metrics['pg'] = avg_metrics.get('acc', 0) + \
                float(torch.mean((masks.view(len(masks), -1)[range(len(masks)), torch.argmax(heatmaps.view(len(heatmaps), -1), 1)] != 0).float()))/len(tr)
    if args.model == 'ProtoPNet' and (epoch+1) % 10 == 0:
        # protopnet has two more stages: in the paper they do this after a few epochs
        model.backbone.eval()
        # (stage2) projection of prototypes
        # we use a deterministic dataloader so that we can access images by index
        # later on
        dl = torch.utils.data.DataLoader(ds_noaug, args.batchsize, pin_memory=True)
        features_per_class = [[] for _ in range(ds.num_classes)]
        indices_per_class = [[] for _ in range(ds.num_classes)]
        for i, (x, _, y) in enumerate(tqdm(dl, 'stage2')):
            x = x.to(device)
            with torch.no_grad():
                z = model.features(model.backbone(x)[-1])
                z_shape = z.shape
                z = torch.flatten(z, 2).permute(0, 2, 1)
            xscale = x.shape[3] // z_shape[3]
            yscale = x.shape[2] // z_shape[2]
            for j, (k, zk) in enumerate(zip(y, z)):
                features_per_class[k].append(zk)
                l = [(i*args.batchsize+j, x*xscale, y*yscale, (x+1)*xscale-1, (y+1)*yscale-1) for y in range(z_shape[2]) for x in range(z_shape[3])]
                indices_per_class[k] += l
        model.patch_prototypes = [[] for _ in range(ds.num_classes)]
        model.image_prototypes = [[] for _ in range(ds.num_classes)]
        model.illustrative_prototypes = [[] for _ in range(ds.num_classes)]
        for k, zk in enumerate(features_per_class):
            zk = torch.cat(zk)
            assert len(zk) == len(indices_per_class[k])
            pk = model.prototype_layer.prototypes[0, k]
            # FIXME: maybe we should not allow the same prototype to be reused
            distances = torch.cdist(zk[None], pk[None])[0]
            ix = torch.argmin(distances, 0)
            with torch.no_grad():  # projection
                model.prototype_layer.prototypes[0, k] = zk[ix]
            for i in ix:
                i, x1, y1, x2, y2 = indices_per_class[k][i]
                image = ds_noaug[i][0]
                patch = image[:, y1:y2, x1:x2]
                model.patch_prototypes[k].append(patch)
                model.image_prototypes[k].append(image)
                image = image.clone()
                color = torch.tensor((1, 0, 0))[:, None, None]  # red
                image[:, y1:y1+2, x1:x2] = color
                image[:, y2:y2+2, x1:x2] = color
                image[:, y1:y2, x1:x1+2] = color
                image[:, y1:y2, x2:x2+2] = color
                model.illustrative_prototypes[k].append(image)
        # (stage3) convex optimization of last layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc_layer.parameters():
            param.requires_grad = True
        for _ in tqdm(range(20), 'stage3'):
            for x, _, y in tr:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                stage3_loss = torch.nn.functional.cross_entropy(pred['class'], y)
                stage3_loss += 1e-4*baseline_protopnet.stage3_loss(model)
                fast_opt.zero_grad()
                stage3_loss.backward()
                fast_opt.step()
                avg_losses['stage3'] = avg_losses.get('stage3', 0) + float(stage3_loss)/len(tr)
        for param in model.parameters():
            param.requires_grad = True
        model.backbone.train()
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {" - ".join(f'{k}={v}' for k, v in avg_losses.items())} - Avg metrics: {" - ".join(f'{k}={v}' for k, v in avg_metrics.items())}')
    if args.visualize:
        with torch.no_grad():
            x, _, y = visualize_batch
            x = x[:4].to(device)
            y = y[:4].to(device)
            pred = model(x)
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (18, 8)
        plt.clf()
        for i in range(4):
            plt.subplot(2, 4, i+1)
            if 'bboxes' in pred:
                utils.draw_bboxes(x[i], pred['bboxes'][i].detach(), pred['scores'][i].detach(), args.nstdev)
            else:
                plt.imshow(x[i].cpu().permute(1, 2, 0))
            plt.title(f"y={y[i]} ŷ={pred['class'][i].argmax()}")
            plt.subplot(2, 4, i+4+1)
            if 'heatmap' in pred:
                utils.draw_heatmap(x[i], pred['heatmap'][i].detach())
        plt.suptitle(f'{args.output[:-4]} epoch={epoch+1}')
        plt.savefig(f'{args.output}-epoch-{epoch+1}.png')
        if args.model == 'ProtoPNet' and (epoch+1) % 10 == 0:
            plt.clf()
            for k in range(2 if args.fast else 10):
                for i, image in enumerate(model.illustrative_prototypes[k]):
                    plt.subplot(2, 5, i+1)
                    plt.imshow(image.permute(1, 2, 0))
                plt.suptitle(f'ProtoPNet prototypes for class {k} epoch={epoch+1}')
                plt.savefig(f'{args.output}-epoch-{epoch+1}-prototypes-k{k}.png')
torch.save(model.cpu(), args.output)
