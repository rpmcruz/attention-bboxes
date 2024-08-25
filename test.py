import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset')
parser.add_argument('--captum')
parser.add_argument('--protopnet', action='store_true')
parser.add_argument('--nstdev', type=float, default=1)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

import torch
from torchvision.transforms import v2
import torchmetrics
import data, metrics, utils
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
ds = getattr(data, args.dataset)
ts = ds('/data/toys', 'test', transforms)
ts = torch.utils.data.DataLoader(ts, 8, num_workers=4, pin_memory=True)

############################# MODEL #############################

if args.protopnet:
    import sys
    sys.path.append('protopnet')
model = torch.load(args.model, map_location=device)

########################### BASELINES ###########################

captum_heatmap = None
if args.captum == 'IntegratedGradients':
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)
    def captum_heatmap(x, y):
        return ig.attribute(x, target=y)
if args.captum == 'GuidedGradCAM':
    from captum.attr import GuidedGradCam
    ggc = GuidedGradCam(model, model.backbone.resnet.layer4)
    def captum_heatmap(x, y):
        return ggc.attribute(x, y)

############################# LOOP #############################

acc = torchmetrics.classification.MulticlassAccuracy(ds.num_classes).to(device)
pg = metrics.PointingGame().to(device)
deg_score = metrics.DegradationScore(model).to(device)
sparsity = metrics.Sparsity().to(device)

model.eval()
for i, (x, mask, y) in enumerate(ts):
    x = x.to(device)
    mask = mask.to(device)
    y = y.to(device)
    with torch.no_grad():
        if args.protopnet:
            pred, dist = model(x)
            heatmap = model.distance_2_similarity(model.prototype_distances(x))
            # ProtoPNet heatmaps are for each prototype
            # - should we use mean or max to merge them?
            heatmap = heatmap.max(1)
            heatmap = (heatmap - heatmap.amin((1, 2), True)) / (heatmap.amax((1, 2), True) - heatmap.amin((1, 2), True))
            pred = {'class': pred, 'heatmap': heatmap}
        else:
            pred = model(x)
        acc.update(pred['class'].argmax(1), y)
        if captum_heatmap != None:
            model.return_dict = False
            heatmap = captum_heatmap(x, y)
            heatmap = heatmap.mean(1)
            # make heatmap 7x7 like the others to make it faster
            heatmap = torch.nn.functional.interpolate(heatmap[:, None], (7, 7), mode='bilinear')[:, 0]
            pred['heatmap'] = heatmap
            model.return_dict = True
        if 'heatmap' in pred:
            pg.update(pred['heatmap'], mask)
            deg_score.update(x, y, pred['heatmap'])
            sparsity.update(pred['heatmap'])
    if args.visualize and i == 0:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (18, 8)
        plt.clf()
        for i in range(4):
            plt.subplot(2, 4, i+1)
            if 'bboxes' in pred:
                utils.draw_bboxes(x[i], pred['bboxes'][i].detach(), pred['scores'][i].detach(), args.nstdev)
            plt.title(f"y={y[i]} ŷ={pred['class'][i].argmax()}")
            if 'heatmap' in pred:
                plt.subplot(2, 4, i+4+1)
                utils.draw_heatmap(x[i], pred['heatmap'][i].detach())
        plt.suptitle(f'{args.model[:-4]}')
        plt.savefig(f'{args.model}.png')
    if args.debug:
        break

if 'OnlyClass' in args.model and args.captum == None:
    print(args.model[:-4], args.dataset, acc.compute().item(), 'nan', 'nan', 'nan', sep=',')
else:
    print(args.model[:-4], args.dataset, acc.compute().item(), pg.compute().item(), deg_score.compute().item(), sparsity.compute().item(), sep=',')
