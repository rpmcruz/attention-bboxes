import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()

import torch
from torchvision.transforms import v2
from time import time
import data, models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA #############################

transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(25, 25),
    v2.ToDtype(torch.float32, True),
])
tr = data.STL10('/data2/toys', 'train', transforms)
tr = torch.utils.data.DataLoader(tr, 32, True, num_workers=4, pin_memory=True)

############################# MODEL #############################

model = getattr(models, args.model)(10)
model.to(device)
opt = torch.optim.Adam(model.parameters(), 1e-4)

############################# LOOP #############################

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    for x, y in tr:
        x = x.to(device)
        y = y.to(device)
        pred, _ = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f} - Avg loss: {avg_loss}')

torch.save(model.cpu(), f'model-{args.model}.pth')