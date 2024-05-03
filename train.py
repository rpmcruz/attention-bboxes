import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
parser.add_argument('output')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--penalty1', type=float, default=10)
parser.add_argument('--penalty2', type=float, default=1)
parser.add_argument('--use-softmax', action='store_true')
args = parser.parse_args()

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

model = getattr(models, args.model)(ds.num_classes, args.use_softmax)
model.to(device)
opt = torch.optim.Adam(model.parameters(), 1e-4)

############################# LOOP #############################

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    avg_acc = 0
    for x, _, y in tr:
        x = x.to(device)
        y = y.to(device)
        pred, scores, bboxes = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        if bboxes != None:
            loss += args.penalty1 * bboxes['gauss_scores'].mean()
            loss += args.penalty2 * bboxes['x_gauss_stdev'].mean()
            loss += args.penalty2 * bboxes['y_gauss_stdev'].mean()
        elif scores != None:
            loss += args.penalty1 * scores.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
        avg_acc += (y == pred.argmax(1)).float().mean() / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss} - Avg acc: {avg_acc}')

torch.save(model.cpu(), args.output)
