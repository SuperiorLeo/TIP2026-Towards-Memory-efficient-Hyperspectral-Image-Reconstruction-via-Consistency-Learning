import os
import os.path
# import h5py
from scipy.io import loadmat
import cv2
import glob
import numpy as np
import argparse
import hdf5storage 
import random


parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--data_path", type=str, default='/home/data/SSROriDataset/NTIRE2020/', help="data path")
parser.add_argument("--patch_size", type=int, default=128, help="data patch size")
parser.add_argument("--stride", type=int, default=64, help="data patch stride")
parser.add_argument("--train_data_path1", type=str, default='/home/data/lengyihong/Dataset/Semi_2020/Test_split', help="preprocess_data_path")
opt = parser.parse_args()


def main():
    if not os.path.exists(opt.train_data_path1):
        os.makedirs(opt.train_data_path1)
    

    process_data(patch_size=opt.patch_size, stride=opt.stride, mode='valid')


def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def process_data(patch_size, stride, mode):
    if mode == 'valid':
        print("\nprocess validing set ...\n")
        patch_num = 1
        filenames_hyper = glob.glob(os.path.join(opt.data_path, 'NTIRE2020_Validation_Spectral', '*.mat'))
        filenames_rgb = glob.glob(os.path.join(opt.data_path, 'NTIRE2020_Validation_Clean', '*.png'))
        filenames_hyper.sort()
        filenames_rgb.sort()
        for k in range(len(filenames_hyper)):

            # list_name.write('{}\n'.format(filenames_hyper[k]))
            print([filenames_hyper[k], filenames_rgb[k]])
            mat = loadmat(filenames_hyper[k]) # 这个是根据mat是用哪个版本保存的决定
            hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [2,0,1])
            hyper = normalize(hyper, max_val=1., min_val=0.)
            # load   rgb image
            rgb = cv2.imread(filenames_rgb[k])  # imread -> BGR model
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.transpose(rgb, [2, 0, 1])
            # print(rgb.shape)
            rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)
            # creat patches
            patches_hyper = Im2Patch(hyper, win=patch_size, stride=stride)
            patches_rgb = Im2Patch(rgb, win=patch_size, stride=stride)
            train_data_path = os.path.join(opt.train_data_path1, 'test'+str(patch_num)+'.mat')
            for j in range(patches_hyper.shape[3]):
                print("generate labeled validing sample #%d" % patch_num)
                sub_hyper = patches_hyper[:, :, :, j]
                sub_rgb = patches_rgb[:, :, :, j]

                train_data_path = os.path.join(opt.train_data_path1, 'test'+str(patch_num)+'.mat')
                hdf5storage.savemat(train_data_path, {'rad': sub_hyper}, format='7.3')
                hdf5storage.savemat(train_data_path, {'rgb': sub_rgb}, format='7.3')

                patch_num += 1


if __name__ == '__main__':
    main()

