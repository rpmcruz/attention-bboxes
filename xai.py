# implementation of xAI methods

import torch

# https://arxiv.org/abs/1512.04150
def CAM(model, layer_act, layer_fc, x, y):
    act = None
    def act_fhook(_, input, output):
        nonlocal act
        act = output
    h = layer_act.register_forward_hook(act_fhook)
    with torch.no_grad():
        model(x)
    h.remove()
    w = layer_fc.weight[y]
    return torch.sum(w[..., None, None]*act, 1)

# https://ieeexplore.ieee.org/document/8237336
def GradCAM(model, layer_act, layer_fc, x, y):
    act = w = None
    def act_fhook(_, input, output):
        nonlocal act
        act = output
    def act_bhook(_, grad_input, grad_output):
        nonlocal w
        w = torch.mean(grad_output[0], (2, 3))
    fh = layer_act.register_forward_hook(act_fhook)
    bh = layer_act.register_full_backward_hook(act_bhook)
    pred = model(x)['class']
    pred = pred[range(len(y)), y].sum()
    pred.backward()
    fh.remove()
    bh.remove()
    # in the paper, they use relu to eliminate the negative values
    # (but maybe we want them to improve our metrics like degredation score)
    return torch.sum(w[..., None, None]*act, 1)

# https://arxiv.org/abs/1704.02685
def DeepLIFT(model, layer_act, layer_fc, x, y):
    baseline = torch.zeros_like(x)
    x.requires_grad = True
    pred_baseline = model(baseline)['class'][range(len(y)), y]
    pred_x = model(x)['class'][range(len(y)), y]
    delta = pred_x - pred_baseline
    delta.sum().backward()
    return torch.mean((x - baseline) * x.grad, 1)

def Occlusion(model, layer_act, layer_fc, x, y):
    occ_w = x.shape[3]//7
    occ_h = x.shape[2]//7
    heatmap = torch.zeros(len(x), 7, 7, device=x.device)
    for occ_i, occ_x in enumerate(range(0, x.shape[3], occ_w)):
        for occ_j, occ_y in enumerate(range(0, x.shape[2], occ_h)):
            occ_img = x.clone()
            occ_img[:, :, occ_x:occ_y+occ_h, occ_x:occ_x+occ_w] = 0
            with torch.no_grad():
                prob = torch.softmax(model(occ_img)['class'], 1)[range(len(y)), y]
                heatmap[:, occ_j, occ_i] = prob
    return heatmap
