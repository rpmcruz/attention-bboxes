import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset')
parser.add_argument('--xai')
parser.add_argument('--protopnet', action='store_true')
parser.add_argument('--nstdev', type=float, default=1)
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
ds = getattr(data, args.dataset)('/data/toys', 'test', transforms, args.crop)
ts = torch.utils.data.DataLoader(ds, args.batchsize, num_workers=4, pin_memory=True)

############################# MODEL #############################

if args.protopnet:
    import sys
    sys.path.append('protopnet')
model = torch.load(args.model, map_location=device, weights_only=False)
model.eval()

########################### BASELINES ###########################

xai = getattr(xai, args.xai) if args.xai else None
if args.protopnet:
    generate_heatmap = lambda x, y: xai(model, model.features.layer4[-1].conv3, model.features.fc, x, y)
else:
    generate_heatmap = lambda x, y: xai(model, model.backbone.resnet.layer4[-1].conv3, model.classifier.output, x, y)

############################# LOOP #############################

acc = torchmetrics.classification.MulticlassAccuracy(ds.num_classes).to(device)
pg = metrics.PointingGame().to(device)
deg_score = metrics.DegradationScore(model).to(device)
sparsity = metrics.Sparsity().to(device)

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
        sparsity.update(pred['heatmap'])
        # degradation score is too slow and requires too much vram, therefore downsample heatmaps that
        # are too large (like those produced by occlusion=image)
        heatmap = pred['heatmap']
        if heatmap.shape[-1] > 7:
            heatmap = torch.nn.functional.interpolate(pred['heatmap'][:, None], (7, 7), mode='bilinear')[:, 0]
        deg_score.update(x, y, heatmap)
    if args.visualize and i == 0:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (18, 8)
        plt.clf()
        for i in range(4):
            plt.subplot(2, 4, i+1)
            if 'bboxes' in pred:
                utils.draw_bboxes(x[i].detach(), pred['bboxes'][i].detach(), pred['scores'][i].detach(), args.nstdev)
            else:
                utils.draw_image(x[i].detach())
            plt.title(f"y={y[i]} ŷ={pred['class'][i].argmax()}")
            if 'heatmap' in pred:
                plt.subplot(2, 4, i+4+1)
                utils.draw_heatmap(x[i].detach(), pred['heatmap'][i].detach())
        plt.suptitle(f'{args.model[:-4]}')
        fname = args.model + '-' + args.xai if args.xai else args.model
        plt.savefig(fname + '.png')

print(args.model[:-4], args.dataset, args.xai, args.crop, acc.compute().item(), pg.compute().item(), deg_score.compute().item(), sparsity.compute().item(), sep=',')
