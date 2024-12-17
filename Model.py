import torch
import torch.nn.functional as F
from torch import nn

from model.autoencoder import Encoder, Decoder
from model.unet import UNetModel
from model.ir_embedder import IR_Embedder
import numpy as np
from model.cross_attention import *

class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor):
        return self.op(x)
    
class Res_fea(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 7, stride=1, padding=3)
        self.op1 = nn.Conv2d(2, 1, 1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor):
        z = self.op(x)
        z_mean = torch.mean(z, 1, keepdim = True)
        z_max, _ = torch.max(z, 1, keepdim = True)
        z = self.op1(torch.cat((z_mean, z_max), 1))
        z = self.sigmoid(z)        
        return z

class ResBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        
        self.in_layers = nn.Sequential(
            nn.Conv2d(channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

        self.res_block = Res_fea(out_channels)
#         if out_channels == channels:
#             self.skip_connection = nn.Identity()
#         else:

        self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor):

        h = self.in_layers(x)
        h = self.out_layers(h)
        
        x = self.skip_connection(x)
        
        att = self.res_block(x-h)
        
        # Add skip ress-attention connection
        return x + h * att

def GaussianDistribution(input):
    mean = 0
    var = 0.01
    noise = np.random.normal(mean, var**0.5, input.shape)
    noise = torch.tensor(noise).cuda()
    image = input + noise
    image = torch.clamp(image, -20, 30)
    return image.float()

# def GaussianDistribution(input):
    
#     mean, log_var = torch.chunk(input, 2, dim=1) #(input, chunk, dim) (输入，拆分的块数，拆分维度)
#     # Clamp the log of variances
#     log_var = torch.clamp(log_var, -30.0, 20.0)
#     # Calculate standard deviation
#     std = torch.exp(0.5 * log_var)
#     return mean, mean + std * torch.randn_like(std)
    
class LDM_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vi_in = nn.Conv2d(1, 16, 3, padding=1)
        self.ir_in = nn.Conv2d(1, 16, 3, padding=1)
        
        self.resnet_down_vi1 = ResBlock(16, 16)
        self.down_vi1 = DownSample(16)
        self.resnet_down_ir1 = ResBlock(16, 16)
        self.down_ir1 = DownSample(16)
        
        self.cross_attention_vi_en_1 = SpatialTransformer(16, 1, 1, 16)
        self.cross_attention_ir_en_1 = SpatialTransformer(16, 1, 1, 16)
        
        self.resnet_down_vi2 = ResBlock(16, 16)
        self.down_vi2 = DownSample(16)
        self.resnet_down_ir2 = ResBlock(16, 16)
        self.down_ir2 = DownSample(16)
        
        self.cross_attention_vi_en_2 = SpatialTransformer(16, 1, 1, 16)
        self.cross_attention_ir_en_2 = SpatialTransformer(16, 1, 1, 16)
        
        self.resnet_down_vi3 = ResBlock(16, 16)
        self.down_vi3 = DownSample(16)
        self.resnet_down_ir3 = ResBlock(16, 16)
        self.down_ir3 = DownSample(16)      
#         self.resnet_down_ir4 = ResBlock(16, 16)
#         self.down_ir4 = DownSample(16)  
        
        self.cross_attention_vi = SpatialTransformer(16, 1, 1, 16)
        self.cross_attention_ir = SpatialTransformer(16, 1, 1, 16)
        
        self.resnet_up_vi1 = ResBlock(32, 16)
        self.up_vi1 = UpSample(16)
        self.resnet_up_ir1 = ResBlock(32, 16)
        self.up_ir1 = UpSample(16)
        
        self.cross_attention_vi_de_1 = SpatialTransformer(16, 1, 1, 16)
        self.cross_attention_ir_de_1 = SpatialTransformer(16, 1, 1, 16)
        
        self.resnet_up_vi2 = ResBlock(32, 16)
        self.up_vi2 = UpSample(16)
        self.resnet_up_ir2 = ResBlock(32, 16)
        self.up_ir2 = UpSample(16)
        
        self.cross_attention_vi_de_2 = SpatialTransformer(16, 1, 1, 16)
        self.cross_attention_ir_de_2 = SpatialTransformer(16, 1, 1, 16)
        
        self.resnet_up_vi3 = ResBlock(32, 16)
        self.up_vi3 = UpSample(16)
        self.resnet_up_ir3 = ResBlock(32, 16)
        self.up_ir3 = UpSample(16)


        self.fusion = nn.Sequential(
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 8),
        )
    
        self.res_fea_vi = Res_fea(16)
        self.res_fea_ir = Res_fea(16)
                    
        self.out = nn.Conv2d(8, 1, 1, padding=0)
        self.tanh = nn.Tanh()
        
    def forward(self, vi_img: torch.Tensor, ir_img: torch.Tensor):
        
     
        
#         vi_encoder_in = self.vi_in(vi_img)
#         ir_encoder_in = self.ir_in(ir_img)

#         vi_encoder = GaussianDistribution(vi_encoder_in)
#         ir_encoder = GaussianDistribution(ir_encoder_in)   
        
#         vi_encoder_n = vi_encoder
#         ir_encoder_n = ir_encoder

        vi_encoder_in = self.vi_in(vi_img)
        ir_encoder_in = self.ir_in(ir_img)

        vi_fea1 = self.resnet_down_vi1(vi_encoder_in)
        vi_fea_down1 = self.down_vi1(vi_fea1)
        ir_fea1 = self.resnet_down_ir1(ir_encoder_in)
        ir_fea_down1 = self.down_ir1(ir_fea1)
        
#         vi_c_fea_down1 = self.cross_attention_vi_en_1(vi_fea_down1, ir_fea_down1)
#         ir_c_fea_down1 = self.cross_attention_ir_en_1(ir_fea_down1, vi_fea_down1)

        vi_fea2 = self.resnet_down_vi2(vi_fea_down1)
        vi_fea_down2 = self.down_vi2(vi_fea2)
        ir_fea2 = self.resnet_down_ir2(ir_fea_down1)
        ir_fea_down2 = self.down_ir2(ir_fea2)
        
        vi_c_fea_down2 = self.cross_attention_vi_en_2(vi_fea_down2, ir_fea_down2)
        ir_c_fea_down2 = self.cross_attention_ir_en_2(ir_fea_down2, vi_fea_down2)
        
        vi_fea3 = self.resnet_down_vi3(vi_c_fea_down2)
        vi_fea_down3 = self.down_vi3(vi_fea3)
        ir_fea3 = self.resnet_down_ir3(ir_c_fea_down2)
        ir_fea_down3 = self.down_ir3(ir_fea3)
#         ir_fea4 = self.resnet_down_ir4(ir_fea_down3)
#         ir_fea_down4 = self.down_ir4(ir_fea4)
        
        cross_fea_vi = self.cross_attention_vi(vi_fea_down3, ir_fea_down3)
        cross_fea_ir = self.cross_attention_ir(ir_fea_down3, vi_fea_down3)        

        vi_encoder = self.up_vi1(cross_fea_vi)
        vi_encoder = self.resnet_up_vi1(torch.cat((vi_encoder, vi_fea3), 1)) # ir_fea3,
        ir_encoder = self.up_ir1(cross_fea_ir)
        ir_encoder = self.resnet_up_ir1(torch.cat((ir_encoder, ir_fea3), 1))        

        vi_c_encoder = self.cross_attention_vi_en_1(vi_encoder, ir_encoder)
        ir_c_encoder = self.cross_attention_ir_en_1(ir_encoder, vi_encoder)
        
        vi_encoder = self.up_vi2(vi_c_encoder)
        vi_encoder = self.resnet_up_vi2(torch.cat((vi_encoder, vi_fea2), 1))        
        ir_encoder = self.up_ir2(ir_c_encoder)
        ir_encoder = self.resnet_up_ir2(torch.cat((ir_encoder, ir_fea2), 1))    
        
#         vi_c_encoder = self.cross_attention_vi_en_2(vi_encoder, ir_encoder)
#         ir_c_encoder = self.cross_attention_ir_en_2(ir_encoder, vi_encoder)
        
        vi_encoder = self.up_vi3(vi_encoder)
        vi_encoder = self.resnet_up_vi3(torch.cat((vi_encoder, vi_fea1), 1))    
        ir_encoder = self.up_ir3(ir_encoder)
        ir_encoder = self.resnet_up_ir3(torch.cat((ir_encoder, ir_fea1), 1))    

        vi_res_att = self.res_fea_vi(vi_encoder_in - vi_encoder)
        ir_res_att = self.res_fea_ir(ir_encoder_in - ir_encoder)
        
        vi_encoder = vi_res_att * vi_encoder
        ir_encoder = ir_res_att * ir_encoder
                    
        img_fusion = self.fusion(torch.cat((vi_encoder, ir_encoder), 1))
     
     #   return self.tanh(self.out(img_fusion)), cross_fea_ir, cross_fea_vi, vi_res_att, ir_res_att, vi_encoder_n, ir_encoder_n
        return self.tanh(self.out(img_fusion)), cross_fea_ir, cross_fea_vi, vi_res_att, ir_res_att