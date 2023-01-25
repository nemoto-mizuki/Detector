import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    def __init__(self, model_cfg, dataset, **kwargs):
        self.cfg = model_cfg
        self.dataset = dataset
        