import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
parser.add_argument('--bbox-confidence', type=float, default=0.5)
parser.add_argument('--top-bboxes', type=int)
parser.add_argument('--deviation-confidence', type=float, default=1.5)
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

import torch
from torchvision.transforms import v2
import torchmetrics
import matplotlib.pyplot as plt
from matplotlib import patches
import data, metrics
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA #############################

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
])
# we are testing with the train-set itself for now
ds = getattr(data, args.dataset)
ts = ds('/data/toys', 'test', transforms)
ts = torch.utils.data.DataLoader(ts, 1, num_workers=4, pin_memory=True)

############################# MODEL #############################

model = torch.load(args.model, map_location=device)

############################# LOOP #############################

acc = torchmetrics.classification.MulticlassAccuracy(ds.num_classes).to(device)
pg = metrics.PointingGame().to(device)

model.eval()
for x, mask, y in ts:
    x = x.to(device)
    mask = mask.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred, scores, bboxes = model(x)
        pred = pred.argmax(1)
        acc.update(pred, y)
        if bboxes != None:
            pg.update(bboxes['spatial_scores'], mask)
    if args.visualize:
        hue = 1-scores
        hue = torch.nn.functional.interpolate(hue, x.shape[-2:], mode='nearest-exact')
        hue = torch.cat((torch.ones_like(hue), hue, hue), 1)
        show = 0.5*x + 0.5*hue
        plt.imshow(show[0].permute(1, 2, 0).cpu())
        xgrid = x.shape[3] // scores.shape[3]
        ygrid = x.shape[2] // scores.shape[2]
        plt.hlines(range(xgrid, x.shape[3], xgrid), 0, x.shape[3], 'green')
        plt.vlines(range(ygrid, x.shape[2], ygrid), 0, x.shape[2], 'green')
        for i in range(scores.shape[2]):
            for j in range(scores.shape[3]):
                plt.text(j*xgrid, i*ygrid, f'{scores[0, 0, i, j]*100:02.0f}', va='bottom')
        if bboxes != None:
            # filter those bounding boxes below a certain level of confidence
            batch_scores = bboxes[f'gauss_scores']
            batch_bboxes = bboxes[f'bboxes_dev{args.deviation_confidence}']
            if args.top_bboxes != None:
                bboxes = [[bboxes[i].cpu() for i in scores.argsort(descending=True)[:5]] for scores, bboxes in zip(batch_scores, batch_bboxes)]
            else:
                bboxes = [[bbox.cpu() for score, bbox in zip(scores, bboxes) if score >= args.bbox_confidence] for scores, bboxes in zip(batch_scores, batch_bboxes)]
            for (x1, y1, x2, y2) in bboxes[0]:
                plt.gca().add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none'))
        plt.title(f'Y={y[0]} Ŷ={pred[0].cpu()}')
        plt.xlim(0, x.shape[3])
        plt.ylim(0, x.shape[2])
        plt.show()

print(args.model, f'{acc.compute().item()*100:.1f}', f'{pg.compute().item()*100:.1f}')
