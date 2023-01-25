import torch
import torch.nn as nn
import math
import time
import torch.functional as F

# t = torch.rand(1, 4, 64, 188, 188).to('cuda')
# u = torch.rand(1, 1, 64, 188, 188)
t = torch.ones(1, 4, 64, 188, 188)
u = torch.ones(1, 1, 64, 188, 188)
t2 = u*2
t3 = u*3
t4 = u*4
t = torch.concat((u,t2,t3,t4), dim=1)
# s = torch.bmm

class Net(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.key = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 1,stride=1),
        )
        self.quely = nn.Sequential(
            nn.Conv3d(4, 4, 1, 1),
        )
        self.value = nn.Sequential(
            nn.Conv3d(4, 4, 1, 1),
        )
        self.dropout = nn.Dropout2d(0.5)
        self.softmax = nn.Softmax(dim=1)
        self.scale = input_channels ** -0.5
    
    def forward(self, t):

        st = time.time()
        k, q, v = self.key(t[:,0,:,:,:]), self.quely(t), self.value(t)
        k = torch.unsqueeze(k, 1)
        dots = torch.einsum('btchw,bkchw->btkhw', (q,k))
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out = torch.mul(attn, v).sum(dim=1, keepdim=True) + t[:,0,...]
        

        en = time.time()
        print(en - st)

        return out
        
dots = torch.einsum('btchw,bkchw->btkhw', (t,u))
print(dots.size())
# net = Net(input_channels=64)
# net.cuda()
# output = net.forward(t)
# print(output)

# conv = nn.Conv3d(4, 4, 1, 1).to('cuda')
# # t = torch.cat(t, dim=1)
# output = conv(t)
# output.size()
