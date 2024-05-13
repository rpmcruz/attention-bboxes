import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('output')
parser.add_argument('--detection', choices=['OneStage', 'FCOS', 'DETR'])
parser.add_argument('--heatmap', choices=['GaussHeatmap', 'LogisticHeatmap'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--penalty', type=float, default=0)
parser.add_argument('--nstdev', type=float, default=1)
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
assert (args.detection == None) == (args.heatmap == None), 'Must enable both or neither detection/heatmap'

import torch
from torchvision.transforms import v2
from time import time
import data, models, utils
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

if detection is None:
    model = models.SimpleModel(backbone, classifier)
elif args.adversarial:
    model = models.ImageOcclusionModel(backbone, classifier, detection, heatmap)
else:
    model = models.EncoderOcclusionModel(backbone, classifier, detection, heatmap)
model.to(device)
opt = torch.optim.Adam(model.parameters())
if args.adversarial:
    adv_opt = torch.optim.Adam(detection.parameters())

############################# LOOP #############################

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    avg_acc = 0
    for x, _, y in tr:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred['class'], y)
        if 'heatmap' in pred:
            loss += args.penalty * pred['heatmap'].mean()
        if args.adversarial:
            adv_loss = torch.nn.functional.cross_entropy(pred['min_class'], y) + \
                -torch.nn.functional.cross_entropy(pred['max_class'], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.adversarial:
            adv_opt.zero_grad()
            adv_loss.backward()
            adv_opt.step()
        avg_loss += float(loss) / len(tr)
        avg_acc += (y == pred.argmax(1)).float().mean() / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss} - Avg acc: {avg_acc}')
    if args.debug:
        utils.draw_bboxes(f'epoch-{epoch+1}-bboxes.png', x[0], pred['bboxes'][0].detach(), args.nstdev)
        utils.draw_heatmap(f'epoch-{epoch+1}-heatmap.png', x[0], pred['heatmap'][0].detach())

torch.save(model.cpu(), args.output)
