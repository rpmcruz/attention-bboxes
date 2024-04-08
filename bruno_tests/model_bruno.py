import torch
import torch.nn as nn
import numpy as np
from utils import * 

class AnchorBackbone(nn.Module):
    def __init__(self) -> list:
        super().__init__()
        
        # What each pixel from an activation maps represents in the image
        self.conv1 = nn.LazyConv2d(128,3,2) # 3x3 original image
        self.conv2 = nn.LazyConv2d(256,3,2) # 7x7 original image
        self.conv3 = nn.LazyConv2d(512,3,2) # 15x15 original image -> ixstride^levels

        # Extracting a 1024 representation from different places of the image
        self.anchor0 = nn.LazyConv2d(1024,3,1) # 17x17 original 
        self.anchor1 = nn.LazyConv2d(1024,1,1) # 15x15 original 
        self.anchor2 = nn.LazyConv2d(1024,(3,1),1) # 17x15 original 
        self.anchor3 = nn.LazyConv2d(1024,(1,3),1) # 15x17 original 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)        

        anc0 = self.anchor0(x)
        anc1 = self.anchor1(x)
        anc2 = self.anchor2(x)
        anc3 = self.anchor3(x)

        return [anc0, anc1, anc2, anc3]
    
# Attention Head
class AttentionHead(nn.Module):
    def __init__(self, N: int = 1) -> None:
        super(AttentionHead, self).__init__()

        # Defining the attention layers
        self.W1 = nn.Linear(in_features=1024, out_features=512)
        self.Wa = nn.Linear(in_features=256, out_features=N)
        self.U = nn.Linear(in_features=512, out_features=256)
        self.V = nn.Linear(in_features=512, out_features=256)

        # Defining the activation functions
        self.act_W1 = nn.ReLU()
        # self.act_Wa = nn.Softmax(dim=1)
        self.act_Wa = nn.Sigmoid()
        self.act_U = nn.Sigmoid()
        self.act_V = nn.Tanh()        

    def forward(self, x: torch.Tensor) -> (torch.Tensor):
        # Reducing the dimension from 1024 to 512        
        h = self.W1(x)
        h = self.act_W1(h)

        # Passing through the attention backbone
        x_1 = self.U(h)
        x_1 = self.act_U(x_1)

        x_2 = self.V(h)
        x_2 = self.act_V(x_2)

        y = torch.mul(x_1, x_2)

        # Defining the importance of the extracted feature
        att = self.Wa(y)
        att = self.act_Wa(att)
        return att, h

class ClusterAttention(nn.Module):
    def __init__(self) -> None:
        super(ClusterAttention, self).__init__()
        self.cluster = nn.Linear(in_features=512, out_features=1)
        self.act_cluster = nn.Sigmoid()
    
    def forward(self,x):
        x = self.cluster(x)
        return self.act_cluster(x)

class AttModel(nn.Module):
    def __init__(self, n_classes = 10, n_clusters = 10) -> None:
        super(AttModel, self).__init__()
        self.n_clusters = n_clusters

        self.classifier = nn.Linear(in_features=512, out_features=n_classes)
        self.act_class = nn.Softmax(dim=1)

        self.backbone = AnchorBackbone()
        self.attention_head = AttentionHead()
        self.attention_cluster = ClusterAttention()        

        self.rep_cluster = torch.zeros((10,512))
        self.att_cluster = torch.zeros((10,1))
        self.box_coords = torch.zeros((10,4))

    def forward(self, x):
        x = self.backbone(x)
        feat, coords = feat_organizer(x)
        feat_clusters = cluster_features(feat, coords, num_clusters=self.n_clusters)

        for i in range(len(feat_clusters)):
            feat_aux = feat_clusters[i]['Features']
            coords_aux = feat_clusters[i]['Projections']
            coords_aux = torch.tensor(np.array(coords_aux), dtype=torch.float32)

            att, h = self.attention_head(feat_aux)
            self.box_coords[i,:] = attention_pooling(tiles_vector=coords_aux, atts=nn.functional.softmax(att, dim=0))
            self.rep_cluster[i,:] = attention_pooling(tiles_vector=h, atts=nn.functional.softmax(att, dim=0))
            self.att_cluster[i,:] = self.attention_cluster(self.rep_cluster[i,:])            
        
        rep_final = attention_pooling(tiles_vector=self.rep_cluster, atts=nn.functional.softmax(self.att_cluster, dim=0))
        logits = self.classifier(rep_final)
        
        return self.act_class(logits), self.att_cluster, self.box_coords