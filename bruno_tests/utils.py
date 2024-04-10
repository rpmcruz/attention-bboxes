import torch
import numpy as np
from torch_kmeans import KMeans
from itertools import compress

# Melhorar essa funcao e entender melhor como é a projeção
def corner_projection(i: int, j: int, kernel_size: tuple) -> tuple:
    if kernel_size[0] == 3:
        w = 17
        i_proj = i*8
    else:
        w = 15
        i_proj = i*8

    if kernel_size[1] == 3:
        h = 17
        j_proj = j*8
    else:
        h = 15
        j_proj = j*8

    return torch.tensor([i_proj, j_proj, w, h], dtype=torch.float32).view(1,-1)

def feat_organizer(features: list) -> list:
    kernel = [(3,3), (1,1), (3,1), (1,3)]    
    feat_1024 = torch.empty(size=(400,1024))
    xywh_proj = torch.empty(size=(400,4))            

    counter = 0
    for ind in range(len(features)):                
        anch = features[ind]                
        x_pos,y_pos = anch.shape[-2],anch.shape[-1]

        for i in range(x_pos):
            for j in range(y_pos):
                feat_1024[counter,:] = anch[:,:,i,j]
                xywh_proj[counter,:] = corner_projection(i,j, kernel_size=kernel[ind])
                counter += 1

    return feat_1024, xywh_proj


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
    
    return feat_groups, proj_groups

def attention_pooling(tiles_vector: torch.Tensor, atts: torch.Tensor) -> torch.Tensor:    
    return torch.matmul(torch.t(atts),tiles_vector)