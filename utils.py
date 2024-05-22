import torch
import matplotlib.pyplot as plt
from matplotlib import patches

def draw_bboxes(output, image, bboxes, nstdev=1):
    plt.imshow(image.cpu().permute(1, 2, 0))
    bboxes = bboxes.cpu()
    # 68.27% of the data falls within 1 stddev of the mean
    # 86.64% of the data falls within 1.5 stddevs of the mean
    # 95.44% of the data falls within 2 stddevs of the mean
    H = W = 1  # bboxes are already in absolute coordinates (i.e., not normalized)
    bx = (bboxes[0]-bboxes[2]/2)*W
    by = (bboxes[1]-bboxes[2]/2)*H
    bw = bboxes[2]*W*nstdev
    bh = bboxes[3]*H*nstdev
    for x, y, w, h in zip(bx, by, bw, bh):
        plt.gca().add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.savefig(output)

def draw_heatmap(output, image, heatmap):
    hue = 1-heatmap
    hue = torch.nn.functional.interpolate(hue[None], image.shape[1:], mode='nearest-exact')[0]
    hue = torch.cat((torch.ones_like(hue), hue, hue), 0)
    show = 0.5*image + 0.5*hue
    print('show:', show.min(), show.max(), 'heatmap:', heatmap.min(), heatmap.max())
    plt.imshow(show.cpu().permute(1, 2, 0))
    plt.savefig(output)