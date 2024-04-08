import torch
import numpy as np
from sklearn.cluster import KMeans
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

    return (i_proj, j_proj, w, h)

def feat_organizer(features: list) -> list:
    kernel = [(3,3), (1,1), (3,1), (1,3)]
    feat_1024 = []
    xywh_proj = []    

    for ind in range(len(features)):                
        anch = features[ind]
        anch = torch.squeeze(anch)
        x_pos,y_pos = anch.shape[-2],anch.shape[-1]

        for i in range(x_pos):
            for j in range(y_pos):
                feat_1024.append(anch[:,i,j])
                xywh_proj.append(corner_projection(i,j, kernel_size=kernel[ind]))

    return feat_1024, xywh_proj

def cluster_features(anchor_feat: list, anchor_proj: list, num_clusters = 10) -> dict:
    feat_list = [] 
    feat_groups = []   
    for feat in anchor_feat:
        feat_list.append(feat.detach().cpu().numpy())
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(np.array(feat_list))
    for i in range(num_clusters):
        aux_feat = list(compress(anchor_feat, kmeans.labels_ == i))
        aux_tensor = aux_feat[0]
        for feat in aux_feat[1:]:
            aux_tensor = torch.vstack((aux_tensor,feat))

        aux_dict = {
            'Features': aux_tensor,
            'Projections': list(compress(anchor_proj, kmeans.labels_ == i))
            }
        feat_groups.append(aux_dict)
    return feat_groups

def attention_pooling(tiles_vector: torch.Tensor, atts: torch.Tensor) -> torch.Tensor:    
    return torch.matmul(torch.t(atts),tiles_vector)