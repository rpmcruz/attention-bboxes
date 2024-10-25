import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('dataset')#, choices=['Birds', 'StanfordCars', 'StanfordDogs'])
parser.add_argument('model', choices=['ViT', 'OnlyClass', 'Heatmap', 'SimpleDet', 'FasterRCNN', 'FCOS', 'DETR'])
parser.add_argument('--heatmap', choices=['none', 'GaussHeatmap', 'LogisticHeatmap'], default='GaussHeatmap')
parser.add_argument('--sigmoid', action='store_true')
parser.add_argument('--l1', type=float, default=0)
parser.add_argument('--nstdev', type=float, default=1)
parser.add_argument('--occlusion', default='encoder', choices=['none', 'encoder', 'image'])
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--fast', action='store_true')
parser.add_argument('--resume')
args = parser.parse_args()

# ugly, but this makes it easier for the script
if args.resume == 'none': args.resume = None

import torch
from torchvision.transforms import v2
from time import time
from tqdm import tqdm
import data, models, utils
import baseline_vit
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

if args.model == 'ViT':
    model = baseline_vit.ViT(ds.num_classes)
    opt = torch.optim.AdamW(model.parameters(), args.lr)
else:
    if args.resume:
        model = torch.load(args.resume, device)
        backbone = model.backbone
        classifier = model.classifier
        detection = model.detection
        heatmap = model.bboxes2heatmap
        occlusion = model.occlusion_level
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
    opt = torch.optim.AdamW([
        {'params': backbone.parameters(), 'lr': args.lr/10},
        {'params': list(classifier.parameters()) + (list(detection.parameters()) if detection != None else [])}
    ], args.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
model.to(device)

############################# LOOP #############################

visualize_batch = next(iter(torch.utils.data.DataLoader(ds_noaug, 4)))

model.train()
for epoch in range(args.epochs):
    # make it smoother by training with a low learning rate and then increase
    if args.model != 'OnlyClass':
        magnitude = 2 if epoch < 10 else 1 if epoch < 20 else 0
        lr = args.lr / 10**magnitude
        if len(opt.param_groups) == 2:
            opt.param_groups[0]['lr'] = lr/10
            opt.param_groups[1]['lr'] = lr
        else:
            opt.param_groups[0]['lr'] = lr

    tic = time()
    avg_losses = {}
    avg_metrics = {}
    for x, masks, y in tr:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred['class'], y)
        if 'heatmap' in pred:
            l1 = torch.mean(pred['heatmap'])
            loss += args.l1 * l1
            avg_losses['l1'] = avg_losses.get('l1', 0) + float(l1)/len(tr)
        if 'bboxes' in pred:
            avg_metrics['bbox_size'] = avg_metrics.get('bbox_size', 0) + float(torch.mean(pred['bboxes'][:, 2:]))/len(tr)
        opt.zero_grad()
        loss.backward(retain_graph=args.adversarial)
        if args.adversarial:
            # temporarily disable gradients for backbone and classifier
            for module in [backbone, classifier]:
                for param in module.parameters():
                    param.requires_grad = False
            adv_loss = -torch.nn.functional.cross_entropy(pred['max_class'], y)
            #torch.nn.functional.cross_entropy(pred['min_class'], y)
            adv_loss.backward()
            for module in [backbone, classifier]:
                for param in module.parameters():
                    param.requires_grad = True
            avg_losses['adv'] = avg_losses.get('adv', 0) + float(adv_loss)/len(tr)
        avg_losses['loss'] = avg_losses.get('loss', 0) + float(loss)/len(tr)
        opt.step()
        avg_metrics['acc'] = avg_metrics.get('acc', 0) + (y == pred['class'].argmax(1)).float().mean()/len(tr)
        if 'heatmap' in pred:
            masks = masks.to(device)
            heatmaps = pred['heatmap'][:, None]
            heatmaps = torch.nn.functional.interpolate(heatmaps, masks.shape[-2:], mode='bilinear')
            avg_metrics['pg'] = avg_metrics.get('acc', 0) + \
                float(torch.mean((masks.view(len(masks), -1)[range(len(masks)), torch.argmax(heatmaps.view(len(heatmaps), -1), 1)] != 0).float()))/len(tr)
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
                plt.imshow(utils.unnormalize(x[i]).cpu())
            plt.title(f"y={y[i]} ŷ={pred['class'][i].argmax()}")
            plt.subplot(2, 4, i+4+1)
            if 'heatmap' in pred:
                utils.draw_heatmap(x[i], pred['heatmap'][i].detach())
        plt.suptitle(f'{args.output[:-4]} epoch={epoch+1}')
        plt.savefig(f'{args.output}-epoch-{epoch+1}.png')

torch.save(model.cpu(), args.output)