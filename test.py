import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib import patches
import data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA #############################

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, True),
])
# we are testing with the train-set itself for now
ts = data.STL10('/data2/toys', 'train', transforms)
ts = torch.utils.data.DataLoader(ts, 1, num_workers=4, pin_memory=True)

############################# MODEL #############################

model = torch.load(args.model, map_location=device)

############################# LOOP #############################

model.eval()
for x, y in ts:
    x = x.to(device)
    with torch.no_grad():
        pred, scores, bboxes = model(x)
        pred = pred.argmax(1)
    hue = 1-scores
    hue = torch.nn.functional.interpolate(hue, (96, 96))
    hue = torch.cat((torch.ones_like(hue), hue, hue), 1)
    show = 0.5*x + 0.5*hue
    plt.imshow(show[0].permute(1, 2, 0).cpu())
    plt.hlines(range(16, 96, 16), 0, show.shape[3], 'green')
    plt.vlines(range(16, 96, 16), 0, show.shape[2], 'green')
    for i in range(scores.shape[2]):
        for j in range(scores.shape[3]):
            plt.text(j*16, i*16, f'{scores[0, 0, i, j]*100:02.0f}', va='top')
    for (x1, y1, x2, y2) in bboxes[0]:
        plt.gca().add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none'))
    plt.title(f'Y={y[0]} Ŷ={pred[0].cpu()}')
    plt.show()
