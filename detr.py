# https://github.com/facebookresearch/detr

import torch
import torchvision

class DETR(torch.nn.Module):
    def __init__(self, num_classes, use_softmax, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(torchvision.models.resnet50(weights='DEFAULT').children())[:-2])
        self.conv = torch.nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = torch.nn.Transformer(hidden_dim, nheads,
        num_encoder_layers, num_decoder_layers)
        self.linear_class = torch.nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = torch.nn.Linear(hidden_dim, 4)
        self.query_pos = torch.nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = torch.nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = torch.nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
        self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()

# TODO: bipartite matching loss