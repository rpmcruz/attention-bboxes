import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('output')
parser.add_argument('--detection', choices=['OneStage', 'FCOS', 'DETR'])
parser.add_argument('--heatmap', choices=['GaussHeatmap'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--penalty', type=float, default=0)
parser.add_argument('--nstdev', type=float, default=1)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
assert (args.detection == None) == (args.heatmap == None), 'Must enable both or neither detection/heatmap'

import torch
from torchvision.transforms import v2
from time import time
import data, models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA #############################

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(0.2, 0.2),
    v2.ToDtype(torch.float32, True),
])
ds = getattr(data, args.dataset)
tr = ds('/data/toys', 'train', transforms)
tr = torch.utils.data.DataLoader(tr, 32, True, num_workers=4, pin_memory=True)

############################# MODEL #############################

backbone = models.Backbone()
classifier = models.Classifier(ds.num_classes)
detection = getattr(models, args.detection)() if args.detection else None
heatmap = getattr(models, args.heatmap)() if args.heatmap else None
model = models.Model(backbone, classifier, detection, heatmap)
model.to(device)
opt = torch.optim.Adam(model.parameters())

############################# LOOP #############################

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    avg_acc = 0
    for x, _, y in tr:
        x = x.to(device)
        y = y.to(device)
        pred, heatmap, bboxes = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        if heatmap != None:
            loss += args.penalty * heatmap.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
        avg_acc += (y == pred.argmax(1)).float().mean() / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss} - Avg acc: {avg_acc}')
    if args.debug:
        import matplotlib.pyplot as plt
        from matplotlib import patches
        plt.imshow(x[0].cpu().permute(1, 2, 0))
        bboxes = bboxes[0].cpu().detach()
        # 68.27% of the data falls within 1 stddev of the mean
        # 86.64% of the data falls within 1.5 stddevs of the mean
        # 95.44% of the data falls within 2 stddevs of the mean
        bx = (bboxes[0]-bboxes[2]/2)*x.shape[3]
        by = (bboxes[1]-bboxes[2]/2)*x.shape[2]
        bw = bboxes[2]*x.shape[3]*args.nstdev
        bh = bboxes[3]*x.shape[2]*args.nstdev
        for x, y, w, h in zip(bx, by, bw, bh):
            plt.gca().add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
        plt.show()

torch.save(model.cpu(), args.output)
