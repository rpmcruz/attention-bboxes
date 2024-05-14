import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('dataset')
parser.add_argument('--detection', choices=['OneStage', 'FCOS', 'DETR'])
parser.add_argument('--heatmap', choices=['GaussHeatmap', 'LogisticHeatmap'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--penalty', type=float, default=0)
parser.add_argument('--nstdev', type=float, default=1)
parser.add_argument('--occlusion', default='encoder', choices=['none', 'encoder', 'image'])
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
tr = torch.utils.data.DataLoader(tr, args.batchsize, True, num_workers=4, pin_memory=True)

############################# MODEL #############################

backbone = models.Backbone()
classifier = models.Classifier(ds.num_classes)
detection = getattr(models, args.detection)() if args.detection else None
heatmap = getattr(models, args.heatmap)() if args.heatmap else None

if detection is None or args.occlusion == 'none':
    model = models.SimpleModel(backbone, classifier)
elif args.occlusion == 'encoder':
    model = models.EncoderOcclusionModel(backbone, classifier, detection, heatmap, args.adversarial)
else:  # image
    model = models.ImageOcclusionModel(backbone, classifier, detection, heatmap, args.adversarial)
model.to(device)
opt = torch.optim.Adam(model.parameters())

############################# LOOP #############################

torch.autograd.set_detect_anomaly(True)

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    avg_adv_loss = 0
    avg_acc = 0
    for x, _, y in tr:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred['class'], y)
        #if 'heatmap' in pred:
        #    loss = loss + (args.penalty * pred['heatmap'].mean())
        opt.zero_grad()
        loss.backward(retain_graph=args.adversarial)
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
            avg_adv_loss += float(adv_loss) / len(tr)
        opt.step()  # <--- erro
        avg_loss += float(loss) / len(tr)
        avg_acc += (y == pred['class'].argmax(1)).float().mean() / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss} - Avg acc: {avg_acc}' + ('Avg adversarial loss: {avg_adv_loss}' if args.adversarial else ''))
    if args.debug:
        utils.draw_bboxes(f'epoch-{epoch+1}-bboxes.png', x[0], pred['bboxes'][0].detach(), args.nstdev)
        utils.draw_heatmap(f'epoch-{epoch+1}-heatmap.png', x[0], pred['heatmap'][0].detach())

torch.save(model.cpu(), args.output)
