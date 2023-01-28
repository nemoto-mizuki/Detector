import torch
import torch.nn as nn



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim) # BatchNorm3d ? どれがいいのか確認する
        self.fn = fn
    def forward(self, x, **kwargs):
        B, T, C, H, W = x.size()
        x = x.contiguous().view(B*T, C, H, W)
        x = self.norm(x).contiguous().view(B, T, C, H, W)
        x = self.fn(x, **kwargs)
        return x

        
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.dim
        hidden_dim = cfg.hidden_dim
        kernel_size = cfg.kernel_size
        stride = cfg.stride
        pad = cfg.pad
        dropout = cfg.dropout
        self.depth = cfg.split_length
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size, stride, pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size, stride, pad),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.contiguous().view(B*T, C, H, W)
        x = self.net(x)
        return x.contiguous().view(B, T, C, H, W)
    

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.dim
        heads = cfg.heads
        dim_head = cfg.dim_head
        kernel_size = cfg.kernel_size
        stride = cfg.stride
        pad = cfg.pad
        dropout = cfg.dropout
        self.split_length = cfg.split_length
        project_out = not (heads == 1 and dim_head == dim)
        out_kernel = cfg.out_conv.kernel_size
        out_stride = cfg.out_conv.stride
        out_pad = cfg.out_conv.pad
        out_drop = cfg.out_conv.dropout  
        self.scale = dim ** -0.5

        self.attend = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(dropout)

        self.to_k = nn.Conv2d(dim, dim_head, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.to_q = nn.Conv2d(dim, dim_head, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.to_v = nn.Conv2d(dim, dim_head, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        # self.token = nn.Parameter(torch.randn(self.split_length))
        self.to_out = nn.Sequential(
            nn.Conv2d(dim_head, dim, kernel_size=out_kernel, stride=out_stride, padding=out_pad),
            nn.Dropout2d(out_drop)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, T, C, H, W = x.size()
        k= self.to_k(x[:,0,:,:,:])
        x = x.contiguous().view(B*T, C, H, W)
        q = self.to_q(x).contiguous().view(B, T, -1, H, W)
        v = self.to_v(x).contiguous().view(B, T, -1, H, W)
        k = torch.unsqueeze(k, dim=1)
        dots = torch.einsum('btchw,bkchw->btkhw', (q,k)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.mul(attn, v).sum(dim=1, keepdim=True)
        out = out.repeat(1, T, 1, 1, 1)
        out = out.contiguous().view(B*T, -1, H, W)
        out = self.to_out(out)
        return out.contiguous().view(B, T, C, H, W)
    # def forward(self, x):
    #     B, T, C, H, W = x.size()
    #     k= self.to_k(x[:,0,:,:,:])
    #     x = x.contiguous().view(B*T, C, H, W)
    #     q = self.to_q(x).contiguous().view(B, T, -1, H, W)
    #     v = self.to_v(x).contiguous().view(B, T, -1, H, W)
    #     k = torch.unsqueeze(k, dim=1)
    #     dots = torch.einsum('btchw,bkchw->btkhw', (q,k)) * self.scale
    #     attn = self.attend(dots)
    #     attn = self.dropout(attn)
    #     # out = torch.matmul(attn, v)
    #     out = torch.mul(attn, v)
    #     out = torch.einsum('btchw,t->btchw',(out,self.token))
    #     out = out.sum(dim=1, keepdim=True)
    #     out = out.repeat(1, T, 1, 1, 1)
    #     out = out.contiguous().view(B*T, -1, H, W)
    #     out = self.to_out(out)
    #     return out.contiguous().view(B, T, C, H, W)    
    
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([])
        dim = cfg.PRENORM.dim
        attention_cfg = cfg.ATTENTION
        feedforward_cfg = cfg.FEEDFORWARD
        for _ in range(cfg.depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(attention_cfg)),
                PreNorm(dim, FeedForward(feedforward_cfg))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        
        return x[:,0,...]
            