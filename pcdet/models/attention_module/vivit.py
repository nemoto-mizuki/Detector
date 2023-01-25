import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vivit_module import Attention, PreNorm, FeedForward
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'mean', in_channels = 64, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = nn.MaxPool2d(3,1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x) # [1,16,3,224,224]
        b, t, n, _ = x.shape

        # cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t) # [1,16,196(14*14),192(16*16*3 -> 192(out_dim))]
        # x = torch.cat((cls_space_tokens, x), dim=2) # [1,16,197,192]
        # x += self.pos_embedding[:, :, :(n + 1)] # [1,16,197,192]
        x = self.dropout(x) # [1,16,197,192]

        x = rearrange(x, 'b t n d -> (b t) n d') #[16, 197, 192]
        x = self.space_transformer(x) #[16, 197, 192]
        x = rearrange(x, '(b t) ... -> b t ...', b=b) #[1,16,192]

        # cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b t n d', b=b, t=t)#[1,1,192]
        # x = torch.cat((cls_temporal_tokens, x), dim=2)#[1,17,192]
        x = rearrange(x, 'b t n d -> (b n) t d')
        x = self.temporal_transformer(x) #[1,17,192]
        x = rearrange(x, '(b n) t d -> b t n d', b=b)
        x = rearrange(x, 'b t (h w) d -> b t d h w',h=47,w=47)
        
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] #[1, 17, 197, 192]

        return self.mlp_head(x)
    
    
    

if __name__ == "__main__":
    
    img = torch.ones([2, 4, 64, 188, 188]).cuda()
    
    model = ViViT(188, 4, 7, 4).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    
    