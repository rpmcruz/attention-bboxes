import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset')
parser.add_argument('--captum')
parser.add_argument('--bbox-confidence', type=float, default=0.5)
parser.add_argument('--top-bboxes', type=int)
parser.add_argument('--deviation-confidence', type=float, default=1.5)
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

import torch
from torchvision.transforms import v2
import torchmetrics
from tqdm import tqdm
import data, metrics, utils
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA #############################

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
])
# we are testing with the train-set itself for now
ds = getattr(data, args.dataset)
ts = ds('/data/toys', 'test', transforms)
ts = torch.utils.data.DataLoader(ts, 8, num_workers=4, pin_memory=True)

############################# MODEL #############################

model = torch.load(args.model, map_location=device)

########################### BASELINES ###########################

baseline_heatmap = None
if args.captum == 'IntegratedGradients':
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)
    def baseline_heatmap(x, y):
        return ig.attribute(x, target=y)
if args.captum == 'GuidedGradCAM':
    from captum.attr import GuidedGradCam
    ggc = GuidedGradCam(model, model.layer4)
    def baseline_heatmap(x, y):
        return ggc.attribute(x, y)

############################# LOOP #############################

acc = torchmetrics.classification.MulticlassAccuracy(ds.num_classes).to(device)
pg = metrics.PointingGame().to(device)
deg_score = metrics.DegradationScore(model).to(device)
entropy = metrics.Entropy().to(device)

model.eval()
for x, mask, y in tqdm(ts):
    x = x.to(device)
    mask = mask.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = model(x)
        acc.update(pred['class'].argmax(1), y)
        if baseline_heatmap != None:
            pred['heatmap'] = baseline_heatmap(x, y)
        if 'heatmap' in pred:
            pg.update(pred['heatmap'], mask)
            deg_score.update(x, y, pred['heatmap'])
            entropy.update(pred['heatmap'])
    if args.visualize:
        utils.draw_bboxes(f'{args.model}-bboxes.png', x[0], pred['bboxes'][0].detach(), args.nstdev)
        utils.draw_heatmap(f'{args.model}-heatmap.png', x[0], pred['heatmap'][0].detach())

print(args.model, f'{acc.compute().item()*100:.1f}', f'{pg.compute().item()*100:.1f}', f'{deg_score.compute().item()*100:.1f}', f'{entropy.compute().item()*100:.1f}')
