import torch
from skimage.draw import rectangle_perimeter

def unnormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)[:, None, None]
    image = image*std + mean
    return image.permute(1, 2, 0)

def draw(image, heatmap, bboxes, scores, nstdev=20):
    heatmap = torch.nn.functional.interpolate(heatmap[None, None], image.shape[:-1], mode='bilinear')[0, 0]
    heatmap = torch.nn.functional.relu(heatmap)
    # normalize heatmap (do the same for scores)
    heatmap = (heatmap-heatmap.min()) / (heatmap.max()-heatmap.min())
    scores = (scores-scores.min()) / (scores.max()-scores.min())
    image = image*heatmap[..., None]
    if bboxes is None:
        return image
    # 68.27% of the data falls within 1 stddev of the mean
    # 86.64% of the data falls within 1.5 stddevs of the mean
    # 95.44% of the data falls within 2 stddevs of the mean
    bboxes = bboxes.cpu() * torch.tensor([image.shape[1], image.shape[0]]*2)[:, None]
    threshold = torch.topk(scores, 4)[0][-1]
    for (cx, cy, sw, sh), score in zip(bboxes.T, scores):
        w, h = sw*nstdev, sh*nstdev
        x = cx - w/2
        y = cy - h/2
        # use threshold or alpha or thickness
        if score >= threshold:
            rr, cc = rectangle_perimeter((y, x), extent=(h, w), shape=image.shape[:-1])
            image[rr, cc] = torch.tensor((1, 0, 0), dtype=torch.float32, device=image.device)
    return image
