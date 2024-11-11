import torch, torchvision

# the implement of ViT of torchvision passes need_weights=False
# but we need the weights to comute rollout attention (heatmap)

def new_forward(self, query, key, value, need_weights):
    return old_forward(self, query, key, value)
old_forward = torch.nn.MultiheadAttention.forward
torch.nn.MultiheadAttention.forward = new_forward

def attention_rollout(As):
    # https://arxiv.org/abs/2005.00928
    rollout = As[0]
    I = torch.eye(As[0].shape[1], device=As[0].device)
    for A in As[1:]:
        rollout = (0.5*A + 0.5*I) @ rollout
    return rollout

class ViT(torch.nn.Module):
    def __init__(self, num_classes, large=False):
        super().__init__()
        if large:
            self.vit = torchvision.models.vit_l_16(weights='DEFAULT')
        else:
            self.vit = torchvision.models.vit_b_16(weights='DEFAULT')
        self.vit.heads.head = torch.nn.LazyLinear(num_classes)
        for layer in self.vit.encoder.layers:
            layer.self_attention.register_forward_hook(self.get_attention_weights)

    def get_attention_weights(self, module, input, output):
        self.attention_weights.append(output[1])

    def forward(self, x):
        self.attention_weights = []
        x = self.vit(x)
        return {'class': x, 'heatmap': attention_rollout(self.attention_weights)}

class ViTb(ViT):
    def __init__(self, num_classes):
        super().__init__(num_classes, False)

class ViTl(ViT):
    def __init__(self, num_classes):
        super().__init__(num_classes, True)
