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
parser.add_argument("--train_data_path1", type=str, default='/home/data/lengyihong/Dataset/Semi_2020', help="preprocess_data_path")
opt = parser.parse_args()


def main():
    if not os.path.exists(opt.train_data_path1):
        os.makedirs(opt.train_data_path1)
    if not os.path.exists(os.path.join(opt.train_data_path1,'Valid')):
        os.makedirs(os.path.join(opt.train_data_path1,'Valid'))
    if not os.path.exists(os.path.join(opt.train_data_path1,'Labeled')):
        os.makedirs(os.path.join(opt.train_data_path1,'Labeled'))
    if not os.path.exists(os.path.join(opt.train_data_path1,'UnLabeled')):
        os.makedirs(os.path.join(opt.train_data_path1,'UnLabeled'))

    process_data(patch_size=opt.patch_size, stride=opt.stride, mode='train')


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
    if mode == 'train':
        print("\nprocess training set ...\n")
        patch_num = 1
        patch_num_valid = 1
        filenames_hyper = glob.glob(os.path.join(opt.data_path, 'NTIRE2020_Train_Spectral', '*.mat'))
        filenames_rgb = glob.glob(os.path.join(opt.data_path, 'NTIRE2020_Train_Clean', '*.png'))
        filenames_hyper.sort()
        filenames_rgb.sort()
        list_name = open(os.path.join(opt.train_data_path1,'data_split_2020.txt'),mode='w')
        # TODO 这里需要先划分10张验证集出来 原本的十张验证集 作为测试集合 剩下的440张图片作为一半labeled 一半unlabeled
        valid = []
        list_name.write('Labeled Valid dataset:\n')
        while len(valid) < 10:
            m = random.randint(0,len(filenames_hyper)-1)
            if m not in valid:
                valid.append(m)
                list_name.write('{}\n'.format(filenames_hyper[m]))
                print([filenames_hyper[m], filenames_rgb[m]])

                mat = loadmat(filenames_hyper[m]) # 这个是根据mat是用哪个版本保存的决定
                hyper = np.float32(np.array(mat['cube']))
                hyper = np.transpose(hyper, [2,0,1])
                hyper = normalize(hyper, max_val=1., min_val=0.)
                # load   rgb image
                rgb = cv2.imread(filenames_rgb[m])  # imread -> BGR model
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = np.transpose(rgb, [2, 0, 1])
                # print(rgb.shape)
                rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)
                # creat patches
                patches_hyper = Im2Patch(hyper, win=patch_size, stride=stride)
                patches_rgb = Im2Patch(rgb, win=patch_size, stride=stride)
                # add data ：重组patches
                for j in range(patches_hyper.shape[3]):
                    print("generate labeled valid sample #%d" % patch_num_valid)
                    sub_hyper = patches_hyper[:, :, :, j]
                    sub_rgb = patches_rgb[:, :, :, j]

                    # train_data_path_array = [opt.train_data_path1, opt.train_data_path2, opt.train_data_path3, opt.train_data_path4]
                    train_data_path_array = [os.path.join(opt.train_data_path1,'Valid')]
                    random.shuffle(train_data_path_array)
                    train_data_path = os.path.join(train_data_path_array[0], 'valid'+str(patch_num_valid)+'.mat')
                    # hdf5storage.savemat(train_data_path, {'rad': sub_hyper}, format='7.3')
                    # hdf5storage.savemat(train_data_path, {'rgb': sub_rgb}, format='7.3')

                    patch_num_valid += 1
            else:
                continue
        seed = []
        
        list_name.write('Labeled Train dataset:\n')
        print(len(filenames_hyper)//2 )
        while len(seed) < (len(filenames_hyper)-10)//2:
            k = random.randint(0,len(filenames_hyper)-1)
            
            # 这里是避免生成重复的数据名称
            if (k not in seed) and (k not in valid):
                seed.append(k)
                list_name.write('{}\n'.format(filenames_hyper[k]))
                print([filenames_hyper[k], filenames_rgb[k]])
                # TODO 这里每个数据集的处理情况都不一样 所以要根据具体情况来看
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
                # add data ：重组patches
                for j in range(patches_hyper.shape[3]):
                    print("generate labeled training sample #%d" % patch_num)
                    sub_hyper = patches_hyper[:, :, :, j]
                    sub_rgb = patches_rgb[:, :, :, j]

                    # train_data_path_array = [opt.train_data_path1, opt.train_data_path2, opt.train_data_path3, opt.train_data_path4]
                    train_data_path_array = [os.path.join(opt.train_data_path1,'Labeled')]
                    random.shuffle(train_data_path_array)
                    train_data_path = os.path.join(train_data_path_array[0], 'train'+str(patch_num)+'.mat')
                    # hdf5storage.savemat(train_data_path, {'rad': sub_hyper}, format='7.3')
                    # hdf5storage.savemat(train_data_path, {'rgb': sub_rgb}, format='7.3')

                    patch_num += 1
            else:
                continue
            
        patch_num_2 = 1
        list_name.write('UnLabeled Train dataset:\n')
        for i in range(len(filenames_hyper)):
            if i not in seed and (i not in valid):
                list_name.write('{}\n'.format(filenames_hyper[i]))
                print([filenames_hyper[i], filenames_rgb[i]])
                # TODO 这里每个数据集的处理情况都不一样 所以要根据具体情况来看
                mat = loadmat(filenames_hyper[i]) # 这个是根据mat是用哪个版本保存的决定
                hyper = np.float32(np.array(mat['cube']))
                hyper = np.transpose(hyper, [2,0,1])
                hyper = normalize(hyper, max_val=1., min_val=0.)
                # load rgb image
                rgb = cv2.imread(filenames_rgb[i])  # imread -> BGR model
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = np.transpose(rgb, [2, 0, 1])
                # print(rgb.shape)
                rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)
                # creat patches
                patches_hyper = Im2Patch(hyper, win=patch_size, stride=stride)
                patches_rgb = Im2Patch(rgb, win=patch_size, stride=stride)
                # add data ：重组patches
                for j in range(patches_hyper.shape[3]):
                    print("generate half unlabeled training sample #%d" % patch_num_2)
                    sub_hyper = patches_hyper[:, :, :, j]
                    sub_rgb = patches_rgb[:, :, :, j]

                    # train_data_path_array = [opt.train_data_path1, opt.train_data_path2, opt.train_data_path3, opt.train_data_path4]
                    train_data_path_array = [os.path.join(opt.train_data_path1,'UnLabeled')]
                    random.shuffle(train_data_path_array)
                    train_data_path = os.path.join(train_data_path_array[0], 'train'+str(patch_num_2)+'.mat')
                    # hdf5storage.savemat(train_data_path, {'rad': sub_hyper}, format='7.3')
                    # hdf5storage.savemat(train_data_path, {'rgb': sub_rgb}, format='7.3')

                    patch_num_2 += 1 

        print("\n Half training set: # samples %d\n" % (patch_num_2-1))


if __name__ == '__main__':
    test = ['0613_0232']
    valid = ['0613_0240']
    unlabeled = ['0311_0152', '0603_0190']
    labeled = ['0603_0187', '0603_0197', '0717_0179']
    in_path = '/home/data/TH3/type7/sea/revise/'
    out_path = '/home/data/TH3/type7/sea/split/'
    main()
    #/home/lengyihong/workspace/ConvMLP_2020/train_data_preprocess2.py
