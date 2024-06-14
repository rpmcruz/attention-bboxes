import torch
import matplotlib.pyplot as plt
from matplotlib import patches

def draw_bboxes(image, bboxes, scores, nstdev=2):
    plt.imshow(image.cpu().permute(1, 2, 0))
    # 68.27% of the data falls within 1 stddev of the mean
    # 86.64% of the data falls within 1.5 stddevs of the mean
    # 95.44% of the data falls within 2 stddevs of the mean
    bboxes = bboxes.cpu() * torch.tensor([image.shape[2], image.shape[1]]*2)[:, None]
    scores = torch.softmax(scores, 0)
    scores = scores/scores.amax()
    for (x, y, sw, sh), score in zip(bboxes.T, scores):
        w, h = sw*nstdev, sh*nstdev
        plt.gca().add_patch(patches.Rectangle((x, y), w, h, alpha=score.item(), linewidth=2, edgecolor='r', facecolor='none'))

def draw_heatmap(image, heatmap):
    heatmap = torch.nn.functional.interpolate(heatmap[None, None], image.shape[1:], mode='bilinear')[0]
    show = image*heatmap
    plt.imshow(show.cpu().permute(1, 2, 0))
