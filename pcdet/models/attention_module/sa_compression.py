import torch
import torch.nn as nn
import numpy as np
from pcdet.utils.self_attention_utils import *
import torch.nn.functional as F


softmax = nn.Softmax(dim=1)

class SelfAttentionCompression(nn.Module):
    def __init__(self, model_cfg, dataset, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.in_channels = model_cfg.IN_CHANNEL
        self.width = model_cfg.WIDTH
        self.height = model_cfg.HEIGHT
        self.time_sequence = model_cfg.SEQUENCE_LENGTH
        self.conv_input = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels//8, (1, 1, 1)),
            nn.BatchNorm3d(self.in_channels//8),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.in_channels = self.in_channels//8
        self.SA_layer1 = SALayer1(self.in_channels, self.width//2, self.height//2)
        self.down_conv = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels, (1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(self.in_channels),
            nn.ReLU()
        )
        self.SA_layer2 = SALayer2(self.in_channels, self.width//2, self.height//2)
        self.pool = nn.AvgPool3d((1, 1, 1),(1, 2, 2))
        # self.out_pool = nn.AvgPool3d((4, 1, 1), (1, 1, 1))
        self.up = nn.Upsample(scale_factor=2)
        self.out_conv = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels*8, kernel_size=(8, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(self.in_channels*8),
            nn.ReLU()
        )
    def forward(self, batch_dict):
        x = batch_dict['spatial_features']  
        x = x.permute(1, 2, 0, 3, 4)
        x = self.conv_input(x)
        x = self.pool(x)
        x = self.SA_layer1(x)
        x = self.down_conv(x)
        x = self.SA_layer2(x)
        x = self.up(x)
        x = self.out_conv(x)
        x = torch.squeeze(x, 2)
        batch_dict['spatial_features'] = x
        

        return batch_dict        
        
class SelfAttentionCompression_v2(nn.Module):
    def __init__(self, model_cfg, dataset, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.in_channels = model_cfg.IN_CHANNEL
        self.width = model_cfg.WIDTH
        self.height = model_cfg.HEIGHT
        self.time_sequence = model_cfg.SEQUENCE_LENGTH
        self.conv_input = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels//8, (1, 1, 1)),
            nn.BatchNorm3d(self.in_channels//8),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.in_channels = self.in_channels//8
        self.SA_layer1 = SALayer1(self.in_channels, self.width//2, self.height//2)
        self.down_conv = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels, (1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(self.in_channels),
            nn.ReLU()
        )
        self.SA_layer2 = SALayer2(self.in_channels, self.width//2, self.height//2)
        self.pool = nn.AvgPool3d((1, 1, 1),(1, 2, 2))
        # self.out_pool = nn.AvgPool3d((4, 1, 1), (1, 1, 1))
        self.up = nn.Upsample(size=(4,188,188))
        self.out = nn.Sequential(nn.Conv3d(32, 256, (1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

    def forward(self, batch_dict):
        x = batch_dict['spatial_features']  #[4,2,256,188,188]
        x = x.permute(1, 2, 0, 3, 4)  #[2,256,4,188,188]
        x = self.conv_input(x)  #[2,32,4,188,188]
        x = self.pool(x)  #[2,32,4,94,94]
        x = self.SA_layer1(x)  #[2,32,4,94,94]
        x = self.down_conv(x)  #[2,32,4,94,94]
        x = self.SA_layer2(x)  #[2,32,4,94,94]
        x = self.up(x)  #[2,32,4,188,188]
        x = self.out(x)
        x = x[:,:,0,...]  # [:,:,-1,...]
        batch_dict['spatial_features'] = x
        

        return batch_dict        
        
class SelfAttentionCompression_v3(nn.Module):
    def __init__(self, model_cfg, dataset, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.in_channels = model_cfg.IN_CHANNEL
        self.width = model_cfg.WIDTH
        self.height = model_cfg.HEIGHT
        self.time_sequence = model_cfg.SEQUENCE_LENGTH
        self.conv_input = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels//8, (1, 1)),
            nn.BatchNorm3d(self.in_channels//8),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)
        self.in_channels = self.in_channels//8
        self.SA_layer1 = SALayer1(self.in_channels, self.width//2, self.height//2)
        self.down_conv = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels, (1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(self.in_channels),
            nn.ReLU()
        )
        self.SA_layer2 = SALayer2(self.in_channels, self.width//2, self.height//2)
        self.pool = nn.AvgPool3d((1, 1, 1),(1, 2, 2))
        # self.out_pool = nn.AvgPool3d((4, 1, 1), (1, 1, 1))
        self.up = nn.Upsample(size=(4,188,188))
        self.out = nn.Sequential(nn.Conv3d(32, 256, (1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

    def forward(self, batch_dict):
        x = batch_dict['spatial_features']  #[4,2,256,188,188]
        x = x.permute(1, 2, 0, 3, 4)  #[2,256,4,188,188]
        x = self.conv_input(x)  #[2,32,4,188,188]
        x = self.pool(x)  #[2,32,4,94,94]
        x = self.SA_layer1(x)  #[2,32,4,94,94]
        x = self.down_conv(x)  #[2,32,4,94,94]
        x = self.SA_layer2(x)  #[2,32,4,94,94]
        x = self.up(x)  #[2,32,4,188,188]
        x = self.out(x)
        x = x[:,:,0,...]  # [:,:,-1,...]
        batch_dict['spatial_features'] = x
        

        return batch_dict             

def Self_Att(query, key, value):
    energy = torch.bmm(query, key)
    attention = softmax(energy)
    out = torch.bmm(value, attention.permute(0,2,1))
    return out

class SALayer1(nn.Module):
    def __init__(self, in_channels, width, height):
        super(SALayer1, self).__init__()
        
        self.value = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1,3,1), stride=(1,1,1), padding=(0,1,0)),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1,1,3), stride=(1,1,1), padding=(0,0,1))
        )

        self.key = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.Conv3d(in_channels//2, in_channels//2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))
        )

        self.query = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.Conv3d(in_channels//2, in_channels//2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))
        )

        self.layer_norm = nn.LayerNorm((in_channels,4,width,height))        
        
    def forward(self, x):
        x1, x2, x3, x4 = x[:,:,0,:,:].unsqueeze(2), x[:,:,1,:,:].unsqueeze(2), x[:,:,2,:,:].unsqueeze(2), x[:,:,3,:,:].unsqueeze(2)
        batch, C, T, w, h = x1.size()

        value_1, value_2 = self.value(x1).flatten(start_dim=2), self.value(x2).flatten(start_dim=2)
        key_1, key_2 = self.key(x1).flatten(start_dim=2), self.key(x2).flatten(start_dim=2)
        query_1, query_2 = self.query(x1).flatten(start_dim=2).permute(0,2,1), self.query(x2).flatten(start_dim=2).permute(0,2,1)

        value_3, value_4 = self.value(x3).flatten(start_dim=2), self.value(x4).flatten(start_dim=2)
        key_3, key_4 = self.key(x3).flatten(start_dim=2), self.key(x4).flatten(start_dim=2)
        query_3, query_4 = self.query(x3).flatten(start_dim=2).permute(0,2,1), self.query(x4).flatten(start_dim=2).permute(0,2,1)

        out_1 = Self_Att(query_2, key_1, value_1).view(batch, C, T, w, h)
        out_2 = Self_Att(query_1, key_2, value_2).view(batch, C, T, w, h)
        out_3 = Self_Att(query_4, key_3, value_3).view(batch, C, T, w, h)
        out_4 = Self_Att(query_3, key_4, value_4).view(batch, C, T, w, h)

        out = self.layer_norm(torch.cat((out_1, out_2, out_3, out_4), dim=2))

        return out + x     
    
      
class SALayer2(nn.Module):
    def __init__(self, in_channels, width, height):
        super(SALayer2, self).__init__()

        self.value = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1,3,1), stride=(1,1,1), padding=(0,1,0)),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1,1,3), stride=(1,1,1), padding=(0,0,1))
        )

        self.key = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.Conv3d(in_channels//2, in_channels//2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))
        )

        self.query = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.Conv3d(in_channels//2, in_channels//2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))
        )

        self.layer_norm = nn.LayerNorm((in_channels,4,width,height))    


    def forward(self, x):
        x1, x2 = x[:,:,0:2,:,:], x[:,:,2:4,:,:]
        batch, C, T, w, h = x1.size()

        value_1, value_2 = self.value(x1).flatten(start_dim=2), self.value(x2).flatten(start_dim=2)
        key_1, key_2 = self.key(x1).flatten(start_dim=2), self.key(x2).flatten(start_dim=2)
        query_1, query_2 = self.query(x1).flatten(start_dim=2).permute(0,2,1), self.query(x2).flatten(start_dim=2).permute(0,2,1)

        out_1 = Self_Att(query_2, key_1, value_1).view(batch, C, T, w, h)
        out_2 = Self_Att(query_1, key_2, value_2).view(batch, C, T, w, h)

        out = self.layer_norm(torch.cat((out_1, out_2), dim=2))
        return out + x
