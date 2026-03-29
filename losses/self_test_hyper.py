

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import hdf5storage
import h5py
from math import exp
import torch.nn.functional as F
from scipy.io import loadmat
from skimage.metrics import structural_similarity

def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)



class Loss_hyper_loss(nn.Module):
    def __init__(self):
        super(Loss_hyper_loss, self).__init__()

    def forward(self,hyper1, hyper2):
        ssim_list = []

        for i in range(31):
            ssim_1 = ssim(hyper1[:,i,:,:],hyper2[:,i,:,:])
            ssim_list.append(ssim_1)
        ssim_tensor = torch.Tensor(ssim_list)
        ssim_all = torch.mean(ssim_tensor)
        loss_ssim_hyper = 1 - ssim_all
        # ssim_value = ssim_all
        return loss_ssim_hyper

class Loss_hyper2(nn.Module):
    def __init__(self):
        super(Loss_hyper2, self).__init__()

    def forward(self,hyper1, hyper2):
        ssim_list = []
        hyper1 = np.float32(hyper1.detach().cpu())
        hyper2 = np.float32(hyper2.detach().cpu())


        for j in range(hyper1.shape[0]):
            for i in range(hyper1.shape[1]):
                ssim_list.append(structural_similarity(hyper1[j,i, :, :], hyper2[j,i, :, :]))

        loss = 1 - np.mean(ssim_list)
        # print(loss)

        return loss

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    # 这里记得加到.cuda()上面
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 

# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
 
# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    # print(img1.size())
    # (_, _, height, width) = img1.size()
    # channel = 1
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
        
    # print(window)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret