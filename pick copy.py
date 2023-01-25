import torch


# a = torch.ones(2*4*3*5*7).view(2*4,3,5,7)
# b = torch.ones(2*4*3*7*5).view(2*4,3,7,5)


# attn = torch.einsum('bchw,bchw->bhw',(a,a))
# attn2 = torch.matmul(a,b)
# attn3 = torch.mul(a,b)
# print(attn.size())

k = torch.ones(1*4*64*188*188).view(1,4,64,188,188)
q = torch.ones(1*4*64*188*188).view(1,4,64,188,188)*0.1
v = torch.ones(1*4*64*188*188).view(1,4,64,188,188)

attn = torch.einsum('btchw,bkchw->btkhw',(q,k))
attn2 = torch.matmul(q,k.permute(0,2,1))
print(attn.size())
