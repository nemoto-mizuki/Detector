import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D


class BEV_Attention(nn.Module):
    def __init__(self, cfg, )