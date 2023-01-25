import torch
import os
import torch.nn as nn
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# k = torch.arange(0, 1*4*32*96*96).view(4,32,96,96).flatten(start_dim=2).permute(0,2,1).cuda()
# q = torch.arange(0, 1*4*32*96*96).view(4,32,96,96).flatten(start_dim=2).cuda()
# v = torch.arange(0, 1*4*32*96*96).view(4,32,96,96).flatten(start_dim=2).cuda()


# dots = torch.bmm(q,k)
# out = torch.mul(dots, v).sum(dim=1, keepdim=True)


class SelfAttention(nn.Module):
    """Self-attention GANにおけるSelf-attention
    ただしSAGANのSpectral Normを抜いているので注意

    Arguments:
        dims {int} -- 4Dテンソルの入力チャンネル
    """    
    def __init__(self, dims):
        super().__init__()
        self.conv_theta = nn.Conv2d(dims, dims // 8, kernel_size=1)
        self.conv_phi = nn.Conv2d(dims, dims // 8, kernel_size=1)
        self.conv_g = nn.Conv2d(dims, dims // 2, kernel_size=1)
        self.conv_attn = nn.Conv2d(dims // 2, dims, kernel_size=1)
        self.sigma_ratio = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, inputs):
        batch, ch, height, width = inputs.size()
        # theta path
        theta = self.conv_theta(inputs)
        theta = theta.view(batch, ch // 8, height * width).permute([0, 2, 1])  # (B, HW, C/8)        
        # phi path
        phi = self.conv_phi(inputs)
        phi = F.max_pool2d(phi, kernel_size=2)  # (B, C/8, H/2, W/2)
        phi = phi.view(batch, ch // 8, height * width // 4)  # (B, C/8, HW/4)
        # attention
        attn = torch.bmm(theta, phi)  # (B, HW, HW/4) 約21GBのGPU RAM 使用量
        attn = F.softmax(attn, dim=-1)
        # g path
        g = self.conv_g(inputs)
        g = F.max_pool2d(g, kernel_size=2)  # (B, C/2, H/2, W/2)
        g = g.view(batch, ch // 2, height * width // 4).permute([0, 2, 1])  # (B, HW/4, C/2)

        attn_g = torch.bmm(attn, g)  # (B, HW, C/2)
        attn_g = attn_g.permute([0, 2, 1]).view(batch, ch // 2, height, width)  # (B, C/2, H, W)
        attn_g = self.conv_attn(attn_g)
        return inputs + self.sigma_ratio * attn_g
    

net = SelfAttention(64).cuda()
input = torch.arange(0., 2*4*64*94*94, dtype=torch.float32).view(2*4,64,94,94).cuda()
output = net.forward(input)
print(output.size())

class SelfAttentionSpatial(nn.Module):
    def __init__(self, dims):
        super().__init__()
        dim = 64
        heads = 6
        dim_head = 128
        kernel_size = 1
        stride = 1
        pad = 0
        dropout =0.2
        self.depth = 6
        project_out = not (heads == 1 and dim_head == dim)
        out_kernel = 3
        out_stride = 1
        out_pad = 1
        out_drop = 0.2  
        self.scale = dim ** -0.5

        self.attend = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(dropout)

        self.to_k = nn.Conv2d(dim, dim_head, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.to_q = nn.Conv2d(dim, dim_head, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.to_v = nn.Conv2d(dim, dim_head, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)

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
        dots = torch.einsum('btchw,bkchw->btkhw', (q,k))
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.mul(attn, v).sum(dim=1, keepdim=True)
        out = out.repeat(1, T, 1, 1, 1)
        out = out.contiguous().view(B*T, -1, H, W)
        out = self.to_out(out)
        return out.contiguous().view(B, T, C, H, W)