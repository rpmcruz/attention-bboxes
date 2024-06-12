# https://arxiv.org/abs/1806.10574
import torch

class PrototypeLayer(torch.nn.Module):
    def __init__(self, num_classes, num_prototypes_per_class, prototype_dim):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        num_prototypes = num_classes*num_prototypes_per_class
        self.prototypes = torch.nn.Parameter(torch.randn(1, num_prototypes, prototype_dim))

    def forward(self, x):
        x = torch.flatten(x, 2).permute(0, 2, 1)  # (B, H*W, F)
        distances = torch.cdist(x, self.prototypes)  # (B, H*W, P)
        print('distances:', distances.shape)
        similarity = torch.log((distances+1)/(distances+1e-7))
        max_similarity = torch.amax(similarity, 1)  # (B, P)
        min_distances, arg_min_distances = torch.min(distances, 1)  # (B, P)
        print('x:', x.shape)
        print('min_distances:', min_distances.shape, 'arg_min_distances:', arg_min_distances.shape)
        min_features = x[range(len(x)), arg_min_distances]
        min_distances_per_class = torch.stack([  # (B, K, P/K)
            min_distances[:, k*self.num_prototypes_per_class:(k+1)*self.num_prototypes_per_class] for k in range(self.num_classes)], 1)
        return max_similarity, min_distances_per_class, min_features

class ProtoPNet(torch.nn.Module):
    def __init__(self, backbone, num_classes, num_prototypes_per_class=10):
        super().__init__()
        self.backbone = backbone
        self.prototype_layer = PrototypeLayer(num_classes, num_prototypes_per_class, 2048)
        self.fc_layer = torch.nn.Linear(num_classes*num_prototypes_per_class, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)[-1]
        max_similarity, min_distances_per_class, min_features = self.prototype_layer(features)
        scores = self.fc_layer(max_similarity)
        return {'class': scores, 'min_distances_per_class': min_distances_per_class,
            'min_features': min_features}

def stage1_loss(min_distances_per_class, y):
    # cross-entropy should also be applied
    num_classes = min_distances_per_class.shape[1]
    ix = torch.nn.functional.one_hot(y, num_classes).bool()
    cluster_cost = torch.mean(torch.amin(min_distances_per_class[ix], 1))
    separation_cost = torch.mean(torch.amin(min_distances_per_class[~ix], 1))
    return cluster_cost - separation_cost

def stage2_projection(model, min_features):
    model.prototype_layer.prototypes.data = min_features

def stage3_loss(model):
    # cross-entropy should also be applied
    c = torch.ones_like(model.fc_layer.weight)
    for k in range(model.num_classes):
        c[k, k*model.num_prototypes_per_class:(k+1)*model.num_prototypes_per_class] = 0
    sparsity_cost = torch.sum(torch.abs(c*model.fc_layer.weight))
    return sparsity_cost