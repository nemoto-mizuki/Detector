import pickle
import pprint
import torch

# with open('/media/Elements/nemoto/OpenPCDet_old/output/media/Elements/nemoto/OpenPCDet/tools/cfgs/waymo_models/centerpoint/default/ckpt/checkpoint_epoch_50.pth', 'rb') as p:
#     l = pickle.load(p)
#     pprint.pprint(l)

weight = torch.load('/media/Elements/nemoto/OpenPCDet_old/output/media/Elements/nemoto/OpenPCDet/tools/cfgs/waymo_models/centerpoint/default/ckpt/checkpoint_epoch_50.pth')
print(weight)