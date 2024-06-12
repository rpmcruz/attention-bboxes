# based on https://github.com/AarohiSingla/Image-Classification-Using-Vision-transformer
import torch

class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()
        self.patcher = torch.nn.Conv2d(in_channels, embedding_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.patcher(x)
        x = torch.flatten(x, 2, 3) 
        return x.permute(0, 2, 1)  # [batch_size, P^2*C, N] -> [batch_size, N, P^2*C]

class MultiheadSelfAttentionBlock(torch.nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.multihead_attn = torch.nn.MultiheadAttention(embedding_dim, num_heads, attn_dropout, batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, attn_output_weights = self.multihead_attn(x, x, x, need_weights=True)
        self.attn_output_weights = attn_output_weights  # required for attention rollout
        return attn_output

class MLPBlock(torch.nn.Module):
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, mlp_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_size, embedding_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(self.layer_norm(x))

class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim, num_heads, attn_dropout)
        self.mlp_block =  MLPBlock(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

class ViT(torch.nn.Module):
    def __init__(self, num_classes, img_size=224, in_channels=3, patch_size=16, num_transformer_layers=12,
        embedding_dim=768, mlp_size=3072, num_heads=12, attn_dropout=0, mlp_dropout=0.1,
        embedding_dropout=0.1):
        super().__init__()
        self.num_patches = (img_size * img_size) // patch_size**2
        self.class_embedding = torch.nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = torch.nn.Parameter(torch.randn(1, self.num_patches+1, embedding_dim))
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.transformer_encoder = torch.nn.Sequential(*[
            TransformerEncoderBlock(embedding_dim, num_heads, mlp_size, mlp_dropout)
            for _ in range(num_transformer_layers)])
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        class_token = self.class_embedding.expand(len(x), -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), 1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        # produce heatmap using attention rollout
        with torch.no_grad():
            A = [t.msa_block.attn_output_weights for t in self.transformer_encoder]
            heatmap = A[0]
            for a in A[1:]:
                I = torch.eye(a.shape[1], device=a.device)
                heatmap = (0.5*a + 0.5*I) @ heatmap
        return {'class': x, 'heatmap': heatmap}
