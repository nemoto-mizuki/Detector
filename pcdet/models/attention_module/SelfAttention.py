import torch
import torch.nn as nn
import numpy as np
from pcdet.utils.self_attention_utils import *
import torch.nn.functional as F
class SelfAttentionTemplate(nn.Module):
    def __init__(self, model_cfg, dataset, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.in_channels = model_cfg.IN_CHANNEL
        self.width = model_cfg.WIDTH
        self.height = model_cfg.HEIGHT
        self.time_sequence = model_cfg.SEQUENCE_LENGTH
        self.softmax = nn.Softmax(dim=-1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=(1,1), padding=(0,0))
        )

        self.value = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            nn.Conv2d(64, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
        )

        self.query = nn.Sequential(
            nn.Conv2d(64, 64//2, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.Conv2d(64//2, 64//2, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )

        self.key1 = nn.Sequential(
            nn.Conv2d(64, 64//2, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.Conv2d(64//2, 64//2, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, self.in_channels, kernel_size=(1,1), padding=(0,0))
        )

        self.pool = nn.AdaptiveAvgPool2d(32)

        self.up = nn.Upsample(188)

        self.key2 = nn.Sequential(
            nn.Conv2d(64, 64//2, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.Conv2d(64//2, 64//2, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )

        self.key3 = nn.Sequential(
            nn.Conv2d(64, 64//2, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.Conv2d(64//2, 64//2, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )


    def SA(self, query, key, value):
        energy = torch.matmul(query, key)
        attention = self.softmax(energy)
        out = torch.matmul(value, attention.permute(0,2,1))
        return out
    
    def forward(self, data_dict):
        input = []
        for i in range(self.time_sequence):
            input.append(data_dict[i]['spatial_features_2d'])
        
        output_add = input[-1]

        for x in range(len(input)):
            input[x] = self.conv1(input[x])
            input[x] = self.pool(input[x])
            
   

        T = len(input)
        batch, C, w, h = input[0].size()
        value_2, query_2 = self.value(input[1]).flatten(start_dim=2), self.query(input[1]).flatten(start_dim=2).permute(0, 2, 1)
        value_3, query_3 = self.value(input[2]).flatten(start_dim=2), self.query(input[2]).flatten(start_dim=2).permute(0, 2, 1)
        value_4, query_4 = self.value(input[3]).flatten(start_dim=2), self.query(input[3]).flatten(start_dim=2).permute(0, 2, 1)
        
        key = self.key1(input[0]).flatten(start_dim=2)
        energy = torch.bmm(query_2, key)
        attention = F.softmax(energy, dim=-1)
        sa_2 = torch.bmm(attention, value_2.permute(0, 2, 1))
        sa_2 = sa_2.permute(0, 2, 1).view(batch, C, w, h)
        # sa_2 = self.up(sa_2)

        key = self.key2(sa_2).flatten(start_dim=2)
        energy = torch.bmm(query_3, key)
        attention = F.softmax(energy, dim=-1)
        sa_2 = torch.bmm(attention, value_3.permute(0, 2, 1))
        sa_2 = sa_2.permute(0, 2, 1).view(batch, C, w, h)

        key = self.key3(sa_2).flatten(start_dim=2)
        energy = torch.bmm(query_4, key)
        attention = F.softmax(energy, dim=-1)
        sa_2 = torch.bmm(attention, value_4.permute(0, 2, 1))
        sa_2 = sa_2.permute(0, 2, 1).view(batch, C, w, h)
        
        sa_2 = self.up(sa_2)
        output = self.conv2(sa_2)
        output += output_add

        del data_dict[:-1]
        torch.cuda.empty_cache()
        data_dict = data_dict[-1]

        data_dict['spatial_features_2d'] = output

        return data_dict



        





