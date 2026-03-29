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

def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
            'epoch': epoch,
            'iter': iteration,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
    
    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)


def record_loss3(loss_csv, epoch, iteration, epoch_time,  loss_ave, loss_sup, sup_hyper_loss, loss_unsup, losses_rgb, losses_cons, test_rmae, test_rmse, test_psnr, val_rmae, val_rmse, val_psnr, lr):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time,  loss_ave, loss_sup, sup_hyper_loss, loss_unsup, losses_rgb, losses_cons, test_rmae, test_rmse, test_psnr, val_rmae, val_rmse, val_psnr, lr))
    loss_csv.flush()    
    loss_csv.close


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse


class Loss_PSNR2(nn.Module):
    def __init__(self):
        super(Loss_PSNR2, self).__init__()

    def forward(self, im_true, im_fake, data_range=1.):
        
        N = im_true.size()[0]
        # C = im_true.size()[1]
        # H = im_true.size()[2]
        # W = im_true.size()[3]
        Itrue = im_true.reshape(N,-1)
        Ifake = im_fake.reshape(N,-1)
        msr = torch.mean((Itrue-Ifake)**2,1)
        psnr = 10. * torch.log(( data_range) / msr) / np.log(10.)
        return torch.mean(psnr)
    

class Loss_train_RMSE(nn.Module):
    def __init__(self):
        super(Loss_train_RMSE, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) 
        rrmse = torch.mean(error.contiguous().view(-1))
        return rrmse
    
class Loss_valid(nn.Module):
    def __init__(self):
        super(Loss_valid, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.contiguous().view(-1))
        return mrae


class Constrast_loss_3(nn.Module):
    def __init__(self, vgg_model):
        super(Constrast_loss_3, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pos, neg, rec):

        filters = hdf5storage.loadmat("/home/data/SSROriData/NTIRE2020/cie_1964_w_gain.mat")['filters']

        filters = torch.Tensor(filters).cuda() 
        shape1 = pos.size() # ([1, 31, 64, 64])
        # pos = pos - pos.min()
        pos_1 = pos.reshape(shape1[0],shape1[1],-1) # ([1, 31, 4096])
        pos_1 = pos_1.permute(0,2,1) # ([1, 4096, 31])
        reRGB_pos = torch.matmul(pos_1,filters)  #torch.Size([1, 4096, 3])
        reRGB_pos = reRGB_pos.permute(0,2,1) # torch.Size([1, 3, 4096])
        reRGB_pos = reRGB_pos.reshape(shape1[0],3,shape1[2],shape1[3]) # torch.Size([1, 3, 64, 64])
        reRGB_pos = reRGB_pos / 255.0
        reRGB_pos = torch.clamp(reRGB_pos, 0, 1)

        neg_1 = neg.reshape(shape1[0],shape1[1],-1)
        neg_1 = neg_1.permute(0,2,1) # ([1, 4096, 31])
        reRGB_neg = torch.matmul(neg_1,filters)  #torch.Size([1, 4096, 3])
        reRGB_neg = reRGB_neg.permute(0,2,1) # torch.Size([1, 3, 4096])
        reRGB_neg = reRGB_neg.reshape(shape1[0],3,shape1[2],shape1[3]) # torch.Size([1, 3, 64, 64])
        reRGB_neg = reRGB_neg / 255.0
        reRGB_neg = torch.clamp(reRGB_neg, 0, 1)

        rec_1 = rec.reshape(shape1[0],shape1[1],-1)
        rec_1 = rec_1.permute(0,2,1) # ([1, 4096, 31])
        reRGB_rec = torch.matmul(rec_1,filters)  #torch.Size([1, 4096, 3])
        reRGB_rec = reRGB_rec.permute(0,2,1) # torch.Size([1, 3, 4096])
        reRGB_rec = reRGB_rec.reshape(shape1[0],3,shape1[2],shape1[3]) # torch.Size([1, 3, 64, 64])
        reRGB_rec = reRGB_rec / 255.0
        reRGB_rec = torch.clamp(reRGB_rec, 0, 1)
        # reRGB_rec = normalize(reRGB_rec, max_val=255., min_val=0.)

        loss_1 = []
        loss_2 = []
        pos_features = self.output_features(reRGB_pos)
        neg_features = self.output_features(reRGB_neg)
        rec_features = self.output_features(reRGB_rec)

        for pos_feature, neg_feature, rec_features in zip(pos_features, neg_features, rec_features):
            loss_1.append(F.mse_loss(pos_feature, rec_features))
            loss_2.append(F.mse_loss(neg_feature, rec_features))
        
        loss_p = sum(loss_1)/len(loss_1)
        loss_n = sum(loss_2)/len(loss_2)

        if loss_n:
            loss = loss_p / loss_n
        else:
            loss = loss_p / (loss_n + 1e-8)

        return loss
    
    
class LossTrainCSS2(nn.Module):
    def __init__(self):
        super(LossTrainCSS2, self).__init__()

    def forward(self, outputs, rgb_label):
        filters = hdf5storage.loadmat("/home/data/SSROriData/NTIRE2020/cie_1964_w_gain.mat")['filters']

        filters = torch.Tensor(filters).cuda() 

        shape1 = outputs.size() # ([1, 31, 64, 64])
   
        outputs_1 = outputs.reshape(shape1[0],shape1[1],-1) # ([1, 31, 4096])
        outputs_1 = outputs_1.permute(0,2,1) # ([1, 4096, 31])
        
        reRGB = torch.matmul(outputs_1,filters)  #torch.Size([1, 4096, 3])

        reRGB = reRGB.permute(0,2,1) # torch.Size([1, 3, 4096])

        reRGB = reRGB.reshape(shape1[0],3,shape1[2],shape1[3]) # torch.Size([1, 3, 64, 64])
        reRGB = reRGB / 255.0
        reRGB = torch.clamp(reRGB, 0, 1)

        rrmse = self.mrae_loss(reRGB, rgb_label)
        # print(type(rrmse))

        return rrmse

    def mrae_loss(self, outputs, label):
        # error = torch.abs(outputs - label) / label
        error = torch.abs(outputs - label) 
        # mrae = torch.mean(error.view(-1))
        mrae = torch.mean(error)
        return mrae
      
class PerpetualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, res, gt):

        filters = hdf5storage.loadmat("/home/data/SSROriData/NTIRE2020/cie_1964_w_gain.mat")['filters']

        filters = torch.Tensor(filters).cuda() 
        shape1 = res.size() # ([1, 31, 64, 64])
        # res = res - res.min()
        res_1 = res.reshape(shape1[0],shape1[1],-1) # ([1, 31, 4096])
        res_1 = res_1.permute(0,2,1) # ([1, 4096, 31])
        reRGB1 = torch.matmul(res_1,filters)  #torch.Size([1, 4096, 3])
        reRGB1 = reRGB1.permute(0,2,1) # torch.Size([1, 3, 4096])
        reRGB1 = reRGB1.reshape(shape1[0],3,shape1[2],shape1[3]) # torch.Size([1, 3, 64, 64])
        reRGB1 = reRGB1 / 255.0
        reRGB1 = torch.clamp(reRGB1, 0, 1)


        gt_1 = gt.reshape(shape1[0],shape1[1],-1)
        gt_1 = gt_1.permute(0,2,1) # ([1, 4096, 31])
        reRGB2 = torch.matmul(gt_1,filters)  #torch.Size([1, 4096, 3])
        reRGB2 = reRGB2.permute(0,2,1) # torch.Size([1, 3, 4096])
        reRGB2 = reRGB2.reshape(shape1[0],3,shape1[2],shape1[3]) # torch.Size([1, 3, 64, 64])
        reRGB2 = reRGB2 / 255.0
        reRGB2 = torch.clamp(reRGB2, 0, 1)


        loss = []
        dehaze_features = self.output_features(reRGB1)
        gt_features = self.output_features(reRGB2)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)
    