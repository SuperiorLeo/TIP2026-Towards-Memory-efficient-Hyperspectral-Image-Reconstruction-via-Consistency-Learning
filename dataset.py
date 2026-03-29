import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
import cv2


class HyperDatasetValid(udata.Dataset):
    def __init__(self, mode='valid'):
        if mode != 'valid':
            raise Exception("Invalid mode!", mode)
        data_path = '/home/data/lyh/Semi_2020/Valid/'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))
        self.keys = data_names
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['rad']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        
        rgb = np.float32(np.array(mat['rgb']))
        
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper

class HyperDatasetTest(udata.Dataset):
    def __init__(self, mode='test'):
        if mode != 'test':
            raise Exception("Intest mode!", mode)
        data_path = '/home/data/lyh/Semi_2020/Test_split/'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))
        self.keys = data_names
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['rad']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        
        rgb = np.float32(np.array(mat['rgb']))
        
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper

class HyperDatasetTrainLabeled(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = '/home/data/lyh/Semi_2020/Labeled/'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['rad']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper
    
class HyperDatasetTrainUnLabeled(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = '/home/data/lyh/Semi_2020/UnLabeled/'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['rad']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrainUnLabeled2(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = '/home/data/lyh/Semi_2020/UnLabeled/'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['rad']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        shape = np.shape(rgb)
        noise = np.float32(np.random.rand(shape[0],shape[1],shape[2]))
        noise_rgb = rgb + 0.1 * noise
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        noise_rgb = np.transpose(noise_rgb, [2,1,0])
        noise_rgb = torch.Tensor(noise_rgb)
        mat.close()
        return rgb, noise_rgb, hyper

class HyperDatasetTrainUnLabeled3(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Unlabel mode!", mode)
        data_path = '/home/data/lyh/Semi_2020/UnLabeled/'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['rad']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        size = hyper.size()
        bank = torch.zeros((size[0], size[1], size[2]))
        bank = torch.Tensor(bank)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper, bank