import torch
import matplotlib.pyplot as plt
from matplotlib import patches

def unnormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)[:, None, None]
    image = image*std + mean
    return image.permute(1, 2, 0)

def draw_bboxes(image, bboxes, scores, nstdev=1):
    plt.imshow(unnormalize(image).cpu())
    # 68.27% of the data falls within 1 stddev of the mean
    # 86.64% of the data falls within 1.5 stddevs of the mean
    # 95.44% of the data falls within 2 stddevs of the mean
    bboxes = bboxes.cpu() * torch.tensor([image.shape[2], image.shape[1]]*2)[:, None]
    scores = torch.softmax(scores, 0)
    scores = scores/scores.amax()
    xlim = (0, image.shape[2])
    ylim = (0, image.shape[1])
    for (cx, cy, sw, sh), score in zip(bboxes.T, scores):
        w, h = sw*nstdev, sh*nstdev
        x = cx - w/2
        y = cy - h/2
        plt.gca().add_patch(patches.Rectangle((x, y), w, h, alpha=score.item(), linewidth=2, edgecolor='r', facecolor='none'))
        xlim = (min(xlim[0], x), max(xlim[1], x+w))
        ylim = (min(ylim[0], y), max(ylim[1], y+h))
    plt.xlim(xlim)
    plt.ylim(ylim[::-1])

def draw_heatmap(image, heatmap):
    heatmap = torch.nn.functional.interpolate(heatmap[None, None], image.shape[1:], mode='bilinear')[0]
    show = image*heatmap
    plt.imshow(unnormalize(show).cpu(), vmin=0, vmax=1)
