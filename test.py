import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset')
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

############################# LOOP #############################

acc = torchmetrics.classification.MulticlassAccuracy(ds.num_classes).to(device)
pg = metrics.PointingGame().to(device)
deg_score = metrics.DegradationScore(model).to(device)

model.eval()
for x, mask, y in tqdm(ts):
    x = x.to(device)
    mask = mask.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = model(x)
        acc.update(pred['class'].argmax(1), y)
        if 'heatmap' in pred:
            pg.update(pred['heatmap'], mask)
            deg_score.update(x, y, pred['heatmap'])
    if args.visualize:
        utils.draw_bboxes(f'{args.model}-bboxes.png', x[0], pred['bboxes'][0].detach(), args.nstdev)
        utils.draw_heatmap(f'{args.model}-heatmap.png', x[0], pred['heatmap'][0].detach())

print(args.model, f'{acc.compute().item()*100:.1f}', f'{pg.compute().item()*100:.1f}', f'{deg_score.compute().item()*100:.1f}')
