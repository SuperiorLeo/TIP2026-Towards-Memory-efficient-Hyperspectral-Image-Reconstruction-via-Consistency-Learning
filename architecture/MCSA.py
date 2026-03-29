# from turtle import forward
from termios import VT1
from tkinter import SEL
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F
import random

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from thop import profile


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ReshapeBlock(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(ReshapeBlock,self).__init__()
        self.conv1 = Conv3x3(inplanes, outplanes, 3, 1)
        self.relu = nn.PReLU()
        self.conv2 = Conv3x3(outplanes, outplanes,3, 1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

class Conv3_1(nn.Module):
    def __init__(self,planes_3,planes_1):
        super(Conv3_1,self).__init__()
        self.conv31 = Conv3x3(planes_3, planes_1, 3, 1)
        self.relu = nn.PReLU()
        self.conv32 = Conv3x3(planes_1, planes_1, 3, 1)
        self.conv1 = nn.Conv2d(planes_1, planes_1, 1, 1, 0, bias=False)

    def forward(self,x):
        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x = self.conv1(x)

        return x
    

class QCO_C(nn.Module):
    def __init__(self,level_num, in_channel):
        super(QCO_C,self).__init__()
        self.conv1 = Conv3x3(in_channel, in_channel, 3, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
        )
        self.level_num = level_num
        self.tran1 = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU()
        )
        self.tran2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU()
        )

    def forward(self,x):
        N,C,H,W = x.shape
        x_1 = self.conv1(x) # torch.Size([1, c, 512, 512])
        # print(x.size())
        x_ave = self.sca(x_1) # torch.Size([1, c, 1, 1])
        # print(x_ave.size()) 
        x_qco = x_ave.view(N,-1)
        x_qco_min,_ = x_qco.min(-1)
        x_qco_min = x_qco_min.unsqueeze(-1)
        x_qco_max,_ = x_qco.max(-1)
        x_qco_max = x_qco_max.unsqueeze(-1)
        q_levels = torch.arange(self.level_num).float().cuda()
        q_levels = q_levels.expand(N, self.level_num)
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (x_qco_max - x_qco_min) + x_qco_min
        q_levels = q_levels.unsqueeze(1)

        # print(q_levels.size())
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0] 
        q_levels_inter = q_levels_inter.unsqueeze(-1) 
        x_qco = x_qco.unsqueeze(-1)
        quant = 1 - torch.abs(q_levels - x_qco)
        quant = quant * (quant > (1 - q_levels_inter)) # [1, C, num]
        # print(quant.size())
        sta = quant.sum(1) 
        sta = sta / (sta.sum(-1).unsqueeze(-1)) 
        sta = sta.unsqueeze(1) #[1,num,1]

        out = sta.unsqueeze(1).permute(0,3,1,2)
        out = torch.cat([out, x_ave],dim=1)
        out = self.tran1(out)

        
        # print(out.size())
        out = self.tran2(out)
        # print(out.size()) # [1, c, 1, 1]

    
        return out

    
class MCSA(nn.Module):

    def __init__(self,inplanes, heads, dim_head, stages):
        super(MCSA,self).__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(stages):
            self.blocks.append(nn.ModuleList([
                Attention(inplanes, heads, dim_head, dropout=0.),
                PreNorm(inplanes,FeedForward(inplanes, inplanes, dropout=0.)),
                PreNorm(inplanes,FeedForward(inplanes, inplanes, dropout=0.))
            ]))

    def forward(self,x):
        b, c, h, w = x.size()

        for (attn, ff1, ff2) in self.blocks:
            out1 = attn(x) + x
            out2 = ff1(out1) + out1
            out3 = ff2(out2) + out1
    
        return out3


class LRMLP2(nn.Module):

    def __init__(self,inplanes,midplanes,outplanes, num_feat):
        super(LRMLP2,self).__init__()
        self.conv0 = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.lred0 = MCSA(midplanes, 2, midplanes//2,num_feat)
        self.lred2 = MCSA(midplanes, 2, midplanes//2,num_feat)
        self.lred1 = MCSA(midplanes*2, 4, midplanes*2//4,num_feat)
        self.out = nn.Conv2d(midplanes, outplanes, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.resp = nn.Conv2d(midplanes*2,midplanes, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.down_sample = nn.Conv2d(midplanes,midplanes*2,4,2,1,bias=False)
        self.up_sample = nn.ConvTranspose2d(midplanes*2,midplanes,4,2,1)


    def forward(self,x):
        f1 = self.conv0(x)
        f2 = self.lred0(f1)

        f3 = self.down_sample(f2)
        f31 = self.lred1(f3)
        f32 = self.up_sample(f31)

        f4 = torch.cat((f2,f32),1)
        f41 = self.resp(f4)
        f42 = self.lred2(f41)

        out = self.out(f42)

        return out
    
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()

        
        self.conv31_down = nn.Conv2d(dim,dim, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.conv31_mid = nn.Conv2d(dim,dim*2, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.conv31_up = nn.Conv2d(dim,dim*3, kernel_size=3, padding=(3 - 1) // 2,bias=False)

        self.num_heads = heads
        self.dim_head = dim_head
        
        self.to_q_h = nn.Linear(dim*3, dim_head * heads * 3, bias=False)
        self.to_k_m = nn.Linear(dim*2, dim_head * heads * 2, bias=False)
        self.to_v_m = nn.Linear(dim*2, dim_head * heads * 2, bias=False)

        self.to_q_l = nn.Linear(dim, dim_head * heads, bias=False)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1)) 
        self.proj_h = nn.Linear(dim_head * heads * 3, dim * 3, bias=True)
        self.proj_m = nn.Linear(dim_head * heads * 2, dim * 2, bias=True)
        self.proj_l = nn.Linear(dim_head * heads, dim, bias=True)

        self.att = QCO_C(dim, dim)
        self.rsp = ReshapeBlock(dim*4, dim)
        self.conv = Conv3_1(dim, dim)

        self.dim = dim

    def forward(self, x):
        """
        x_in: [b,c,h,w]
        return out: [b,h,w,c]
        """

        x_CL = self.conv31_down(x).permute(0, 2, 3, 1)
        x_CM = self.conv31_mid(x).permute(0, 2, 3, 1)
        x_CH = self.conv31_up(x).permute(0, 2, 3, 1)
    
        b, h, w, cl = x_CL.shape

        x_L = x_CL.reshape(b,h*w,cl)
        x_M = x_CM.reshape(b,h*w,cl*2)
        x_H = x_CH.reshape(b,h*w,cl*3)
        
        # TODO low
        q_inp_l = self.to_q_l(x_L)
        q_l = rearrange(q_inp_l, 'b n (h d) -> b h n d', h=self.num_heads)
        q_l = q_l.transpose(-2, -1)
        q_l = F.normalize(q_l, dim=-1, p=2)

        # TODO mid
        k_inp_m = self.to_k_m(x_M)
        v_inp_m = self.to_v_m(x_M)
        k_m, v_m = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                ( k_inp_m, v_inp_m))
        v_m = v_m
        k_m = k_m.transpose(-2, -1)
        v_m = v_m.transpose(-2, -1)
        k_m = F.normalize(k_m, dim=-1, p=2)

        # TODO high
        q_inp_h = self.to_q_h(x_H)        
        q_h = rearrange(q_inp_h, 'b n (h d) -> b h n d', h=self.num_heads)
        q_h = q_h.transpose(-2, -1)
        q_h = F.normalize(q_h, dim=-1, p=2)

        attn_m_l = (q_l @ k_m.transpose(-2, -1)) 
        attn_m_l = attn_m_l * self.rescale
        attn_m_l = attn_m_l.softmax(dim=-1)

        attn_m_h = (q_h @ k_m.transpose(-2, -1)) 
        attn_m_h = attn_m_h * self.rescale
        attn_m_h = attn_m_h.softmax(dim=-1)

        x_l = attn_m_l @ v_m # 4 3 hw # 12
        x_h = attn_m_h @ v_m # 4 12 hw # 48


        x_l = x_l.permute(0, 3, 1, 2)    # Transpose
        x_l = x_l.reshape(b, h * w, self.num_heads * self.dim_head )
        
        out_cl = self.proj_l(x_l).view(b, cl, h, w)

        x_h = x_h.permute(0, 3, 1, 2)    # Transpose
        x_h = x_h.reshape(b, h * w, self.num_heads * self.dim_head * 3)
        
        out_ch = self.proj_h(x_h).view(b, cl*3, h, w)

        H_out = torch.cat((out_cl,out_ch),1)
        H_out = self.rsp(H_out)
        att_before = self.att(x_CL.permute(0, 3, 1, 2))
        H_add = self.conv(H_out)
        
        out = H_add + H_out * att_before
        # print(out.shape)

        return out
    


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x.permute(0, 2, 3, 1)), **kwargs).permute(0, 3, 1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # b, c, h, w = x.shape
        # x = x.reshape(b,c,h*w)
        x = self.net(x)
        # x = x.reshape(b,c,h,w)
        return x



if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    input_tensor = torch.rand(1, 3, 256, 256).cuda()
    
    model = LRMLP2(3,31,31,3).cuda()

    with torch.no_grad():
        output_tensor = model(input_tensor)
    macs, params = profile(model, inputs=(input_tensor, ))
    # print(output_tensor.shape)

    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print('Parameters number is {}; Flops: {}'.format(params,macs))
    print(torch.__version__)
    '''
    Parameters number is  2965595
    Parameters number is 2897867.0; Flops: 94054748092.0
    '''