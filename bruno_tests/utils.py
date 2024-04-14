from itertools import compress

import numpy as np
import torch
from torch_kmeans import KMeans
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Melhorar essa funcao e entender melhor como é a projeção
def corner_projection(i: int, j: int, kernel_size: tuple, strides: tuple) -> tuple:
    w = receptive_field(kernel_size=kernel_size[0], strides=strides)
    h = receptive_field(kernel_size=kernel_size[1], strides=strides)
    i_proj = i*np.prod(strides)
    j_proj = j*np.prod(strides)

    return torch.tensor([i_proj, j_proj, w, h], dtype=torch.float32).view(1,-1)

def receptive_field(kernel_size, strides):
    count = 0
    for kernel,stride in zip(kernel_size, strides):
        if not count:
            img_proj = kernel
        else:
            img_proj = kernel + stride*(img_proj-1)
        count += 1
    return img_proj

def feat_organizer(features: list) -> list:
    kernel = [((3,3,3,3), (3,3,3,3)), ((3,3,3,1), (3,3,3,1)), ((3,3,3,3), (3,3,3,1)), ((3,3,3,1), (3,3,3,3))]
    strides = (2,2,2,1)
    feat_1024 = torch.empty(size=(400,1024))
    xywh_proj = torch.empty(size=(400,4))

    counter = 0
    for ind in range(len(features)):
        anch = features[ind]
        x_pos,y_pos = anch.shape[-2],anch.shape[-1]

        for i in range(x_pos):
            for j in range(y_pos):
                feat_1024[counter,:] = anch[:,:,i,j]
                xywh_proj[counter,:] = corner_projection(i,j, kernel_size=kernel[ind], strides=strides)
                counter += 1

    return feat_1024.to(device), xywh_proj.to(device)

def cluster_features(anchor_feat: torch.Tensor, att_feat: torch.Tensor, anchor_proj: torch.Tensor, num_clusters = 10) -> dict:
    feat_groups = torch.empty(size=(num_clusters, 512))
    proj_groups = torch.empty(size=(num_clusters, 4))

    kmeans = KMeans(n_clusters=num_clusters, verbose=False)
    result = kmeans(anchor_feat[None,:,:])

    for i in range(num_clusters):
        aux_tensor = anchor_feat[torch.squeeze(result.labels == i),:]
        aux_att = att_feat[torch.squeeze(result.labels == i),:]
        aux_proj = anchor_proj[torch.squeeze(result.labels == i),:]

        feat_groups[i,:] = attention_pooling(tiles_vector=aux_tensor, atts=torch.nn.functional.softmax(aux_att, dim=0))
        proj_groups[i,:] = attention_pooling(tiles_vector=aux_proj, atts=torch.nn.functional.softmax(aux_att, dim=0)) # Possivel fonte de erro

    return feat_groups.to(device), proj_groups.to(device)

def attention_pooling(tiles_vector: torch.Tensor, atts: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.t(atts),tiles_vector)

def visualize_boxes(img: torch.Tensor, bboxes: torch.Tensor, atts: torch.Tensor, n_boxes = 3):  
    atts = torch.squeeze(atts)
    values, indices = torch.topk(atts, n_boxes, dim=0)    

    bboxes = bboxes[indices]
    atts = atts[indices]

    atts_np = atts.detach().cpu().numpy()
    atts_str = []
    for aux_att in atts_np:
        atts_str.append(str(np.format_float_positional(aux_att))[0:5])

    # Plotting colors
    green = (0,255,0)
    yellow = (255,255,0)
    red = (255,0,0)

    colors = []
    for i in range(1,n_boxes+1):
        if i < n_boxes/3:
            colors.append(green)
        elif i < 2*n_boxes/3:
            colors.append(yellow)
        else:
            colors.append(red)

    img_with_box = draw_bounding_boxes(image=torch.squeeze(img*255).to(torch.uint8), boxes=box_convert(bboxes, in_fmt = 'xywh', out_fmt = 'xyxy'), labels=atts_str, colors=colors,width=1)
    PIL_image = F.to_pil_image(img_with_box)
    plt.imshow(PIL_image)
    plt.show()
    plt.waitforbuttonpress(0)
    plt.close()

def decode_predictions(y: torch.Tensor):
    if y == 0:
        return 'Airplane'
    elif y == 1:
        return 'Bird'
    elif y == 2:
        return 'Car'
    elif y == 3:
        return 'Cat'
    elif y == 4:
        return 'Deer'
    elif y == 5:
        return 'Dog'
    elif y == 6:
        return 'Horse'
    elif y == 7:
        return 'Monkey'
    elif y == 8:
        return 'Ship'
    else:
        return 'Truck'