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
args = parser.parse_args()

import torch
from torchvision.transforms import v2
from time import time
import data, models, utils
import baseline_protopnet, baseline_vit
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA #############################

train_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(0.2, 0.2),
    v2.ToDtype(torch.float32, True),
])
test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
])
ds = getattr(data, args.dataset)('/data/toys', 'train', train_transforms)
ds_noaug = getattr(data, args.dataset)('/data/toys', 'train', test_transforms)
tr = torch.utils.data.DataLoader(ds, args.batchsize, True, num_workers=4, pin_memory=True)

############################# MODEL #############################

if args.model == 'ProtoPNet':
    backbone = models.Backbone()
    model = baseline_protopnet.ProtoPNet(backbone, ds.num_classes)
    slow_opt = torch.optim.Adam(backbone.parameters(), args.lr/10)
    fast_opt = torch.optim.Adam(model.prototype_layer.parameters(), args.lr)
    late_opt = torch.optim.Adam(model.fc_layer.parameters(), args.lr)
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

visualize_batch = next(iter(tr))

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_losses = {}
    avg_metrics = {}
    for x, masks, y in tr:
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
    if args.model == 'ProtoPNet':
        # protopnet has two more stages: in the paper they do this after a few epochs
        model.backbone.eval()
        # (stage2) projection of prototypes
        # we use a deterministic dataloader so that we can access images by index
        # later on
        dl = torch.utils.data.DataLoader(ds_noaug, args.batchsize, pin_memory=True)
        features_per_class = [[] for _ in range(ds.num_classes)]
        indices_per_class = [[] for _ in range(ds.num_classes)]
        for i, (x, _, y) in enumerate(dl):
            x = x.to(device)
            with torch.no_grad():
                z = model.features(model.backbone(x)[-1])
                z_shape = z.shape
                z = torch.flatten(z, 2).permute(0, 2, 1)
            xscale = x.shape[3] // z_shape[3]
            yscale = x.shape[2] // z_shape[2]
            for j, (k, zk) in enumerate(zip(y, z)):
                features_per_class[k].append(zk)
                indices_per_class[k] += [(i*args.batchsize+j, x*xscale, y*yscale, (x+1)*xscale, (y+1)*yscale) for y in range(z_shape[2]) for x in range(z_shape[3])]
        assert len(features_per_class[0]) == len(indices_per_class[0])
        model.image_prototypes = [[] for _ in range(ds.num_classes)]
        for k, zk in enumerate(features_per_class):
            zk = torch.cat(zk)
            pk = model.prototype_layer.prototypes[0, k]
            print('zk:', zk.shape, 'pk:', pk.shape)
            distances = torch.cdist(zk[None], pk[None])[0]
            ix = torch.argmin(distances, 0)
            print('ix:', ix.shape)
            with torch.no_grad():  # projection
                model.prototype_layer.prototypes[0, k] = zk[ix]
            for i in ix:
                i, x1, y1, x2, y2 = indices_per_class[k][i]
                patch = ds[i][:, y1:y2, x1:x2]
                model.image_prototypes[k].append(patch)
        # (stage3) convex optimization of last layer
        for x, _, y in tr:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            stage3_loss = torch.nn.functional.cross_entropy(pred['class'], y) + baseline_protopnet.stage3_loss(model)
            late_opt.zero_grad()
            stage3_loss.backward()
            late_opt.step()
            avg_losses['stage3'] = avg_losses.get('stage3', 0) + float(stage3_loss)/len(tr)
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
            plt.title(f"y={y[i]} ŷ={pred['class'][i].argmax()}")
            plt.subplot(2, 4, i+4+1)
            if 'heatmap' in pred:
                utils.draw_heatmap(x[i], pred['heatmap'][i].detach())
        plt.suptitle(f'{args.output[:-4]} epoch={epoch+1}')
        plt.savefig(f'{args.output}-epoch-{epoch+1}.png')
        if args.protopnet:
            y = visualize_batch[1][0]
            for patch in model.image_prototypes[y]:
                plt.subplot(2, 5, i+1)
                plt.imshow(patch.permute(1, 2, 0))
            plt.suptitle(f'ProtoPNet prototypes for class {y} epoch={epoch+1}')
            plt.savefig(f'{args.output}-epoch-{epoch+1}-prototypes.png')
torch.save(model.cpu(), args.output)
