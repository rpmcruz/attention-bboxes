import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset')
parser.add_argument('--xai')
parser.add_argument('--protopnet', action='store_true')
parser.add_argument('--crop', action='store_true')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

import torch
from torchvision.transforms import v2
import torchmetrics
import data, metrics, utils, xai
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

############################# DATA #############################

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
# we are testing with the train-set itself for now
ts = ds = getattr(data, args.dataset)('/data/toys', 'test', transforms, args.crop)
if args.visualize:
    # limit to 10 classes and the first example
    limit = [None]*10
    for i in range(len(ds)):
        label = ds.__getitem__(i, True)
        if label < len(limit) and limit[label] is None:
            limit[label] = i
    ts = torch.utils.data.Subset(ts, limit)
ts = torch.utils.data.DataLoader(ts, 1 if args.visualize else args.batchsize, num_workers=4, pin_memory=True)

############################# MODEL #############################

if args.protopnet:
    import sys
    sys.path.append('protopnet')
model = torch.load(args.model, map_location=device, weights_only=False)
model.eval()

########################### BASELINES ###########################

generate_heatmap = None
if args.xai:
    xai = getattr(xai, args.xai)
    if args.protopnet:
        generate_heatmap = lambda x, y: xai(model, model.features.layer2, model.features.layer4[-1].conv3, model.features.fc, x, y)
    else:
        generate_heatmap = lambda x, y: xai(model, model.backbone.resnet.layer2, model.backbone.resnet.layer4[-1].conv3, model.classifier.output, x, y)

############################# LOOP #############################

acc = torchmetrics.classification.MulticlassAccuracy(ds.num_classes).to(device)
deg_score = metrics.DegradationScore(model).to(device)
pg = metrics.PointingGame().to(device)
sparsity = [metrics.Density().to(device), metrics.TotalVariance().to(device), metrics.Entropy().to(device)]

for i, (x, mask, y) in enumerate(ts):
    x = x.to(device)
    mask = mask.to(device)
    y = y.to(device)
    if args.protopnet:
        with torch.no_grad():
            pred, dist = model(x)
        # this is a heatmap for each prototype: (N, K, 7, 7)
        heatmap = model.distance_2_similarity(model.prototype_distances(x))
        # get only the heatmaps of the respective prototype class
        heatmap = torch.cat([heatmap[[i], y[i]*10:(y[i]+1)*10] for i in range(len(y))])
        # get the maximum activation across those prototypes
        heatmap = torch.nn.functional.relu(heatmap.amax(1))
        pred = {'class': pred, 'heatmap': heatmap}
    else:
        with torch.no_grad():
            pred = model(x)
    acc.update(pred['class'].argmax(1), y)
    if generate_heatmap != None:
        pred['heatmap'] = generate_heatmap(x, y)
    if 'heatmap' in pred:
        pg.update(pred['heatmap'], mask)
        for s in sparsity:
            s.update(pred['heatmap'])
        # degradation score is too slow and requires too much vram, therefore downsample heatmaps that
        # are too large (like those produced by occlusion=image)
        heatmap = pred['heatmap']
        if heatmap.shape[-1] > 7:
            heatmap = torch.nn.functional.interpolate(pred['heatmap'][:, None], (7, 7), mode='bilinear')[:, 0]
        deg_score.update(x, y, heatmap)
    if args.visualize:
        from skimage.io import imsave
        x = utils.unnormalize(x[0].detach())
        imsave(f'image-{args.dataset}-{i:02d}.png', (x*255).cpu().type(torch.uint8))
        if 'bboxes' in pred:
            x = utils.draw(x.detach(), pred['heatmap'][0].detach(), pred['bboxes'][0].detach(), pred['scores'][0].detach())
        else:
            x = utils.draw(x.detach(), pred['heatmap'][0].detach(), None, None)
        name = args.model
        if args.xai != None:
            name += '-' + args.xai
        imsave('image-' + name + f'-{i:02d}.png', (x*255).cpu().type(torch.uint8))

print(args.model[:-4], args.dataset, args.xai, args.crop, acc.compute().item(), deg_score.compute().item(), pg.compute().item(), *[s.compute().item() for s in sparsity], sep=',')