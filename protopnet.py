import torch

class PrototypeLayer(torch.nn.Module):
    def __init__(self, num_classes, num_prototypes_per_class, prototype_dim):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        num_prototypes = num_classes*num_prototypes_per_class
        self.prototypes = torch.nn.Parameter(torch.randn(1, num_prototypes, prototype_dim))

    def forward(self, x):
        x = torch.flatten(x, 2).permute(0, 2, 1)
        distances = torch.cdist(x, self.prototypes)  # (B, F, P)
        similarity = torch.log((distances+1)/(distances+1e-7))
        max_similarity = torch.amax(similarity, 1)  # (B, P)
        min_distances = torch.amin(distances, 1)  # (B, P)
        min_distances_per_class = torch.stack([  # (B, K, P/K)
            min_distances[:, k*self.num_prototypes_per_class:(k+1)*self.num_prototypes_per_class] for k in range(self.num_classes)], 1)
        return max_similarity, min_distances_per_class

class ProtoPNet(torch.nn.Module):
    def __init__(self, backbone, num_classes, num_prototypes_per_class=10):
        super().__init__()
        self.backbone = backbone
        self.prototype_layer = PrototypeLayer(num_classes, num_prototypes_per_class, 2048)
        self.fc_layer = torch.nn.Linear(num_classes*num_prototypes_per_class, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)
        max_similarity, min_distances_per_class = self.prototype_layer(features)
        scores = self.fc_layer(max_similarity)
        return {'class': scores, 'min_distances_per_class': min_distances_per_class}

def stage1_loss(min_distances_per_class, y, model):
    num_classes = min_distances_per_class.shape[1]
    num_prototypes_per_class = min_distances_per_class.shape[2]
    ix = torch.nn.functional.one_hot(y, num_classes).bool()
    cluster_cost = torch.mean(torch.amin(min_distances_per_class[ix], 1))
    separation_cost = torch.mean(torch.amin(min_distances_per_class[~ix], 1))
    # in the paper, this is done as a separate step
    c = torch.ones_like(model.fc_layer.weight)
    for k in range(num_classes):
        c[k, k*num_prototypes_per_class:(k+1)*num_prototypes_per_class] = 0
    sparsity_cost = torch.sum(torch.abs(c*model.fc_layer.weight))
    return cluster_cost - separation_cost + sparsity_cost

def stage2_projection(min_distances_per_class):
    pass