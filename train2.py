import os
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my import
from dataset import HyperDatasetTrainLabeled, HyperDatasetTrainUnLabeled3, HyperDatasetValid, HyperDatasetTest
import torch.backends.cudnn as cudnn
from domain_adaption.github.architecture.MCSA import LRMLP2
from utils import *

from trainer7 import Trainer

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(gpu, args):
    cudnn.benchmark = True
    args.local_rank = gpu
    train_data_label = HyperDatasetTrainLabeled(mode='train')
    train_data_unlabel = HyperDatasetTrainUnLabeled3(mode='train')
    val_data = HyperDatasetValid(mode='valid')
    test_data = HyperDatasetTest(mode='test')

    paired_loader = DataLoader(train_data_label, batch_size=args.train_batchsize)
    unpaired_loader = DataLoader(train_data_unlabel, batch_size=args.train_batchsize)
    val_loader = DataLoader(val_data, batch_size=args.val_batchsize) # val 的 batch一般都是1
    test_loader = DataLoader(test_data, batch_size=args.test_batchsize) # val 的 batch一般都是1
    print('there are total %s batches for train' % (len(paired_loader)))
    print('there are total %s batches for val' % (len(val_loader)))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    print('there are total %s batches for val' % (len(test_loader)))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    # create model

    net = LRMLP2(3,31,31,3)
    ema_net = LRMLP2(3,31,31,3)
    test_net = LRMLP2(3,31,31,3)

    ema_net = create_emamodel(ema_net)
    print('student model params: %d' % count_parameters(net))
    # tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    trainer = Trainer(model=net, tmodel=ema_net, cmodel=test_net, args=args, supervised_loader=paired_loader,
                      unsupervised_loader=unpaired_loader,
                      val_loader=val_loader, test_loader = test_loader, iter_per_epoch=len(unpaired_loader), writer=writer)

    trainer.train()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    # parser.add_argument('-g', '--gpus', default=0, type=int, metavar='N')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--train_batchsize', default=15, type=int, help='train batchsize')
    parser.add_argument('--val_batchsize', default=4, type=int, help='val batchsize')
    parser.add_argument('--test_batchsize', default=4, type=int, help='test batchsize')
    parser.add_argument('--resume', default='', type=str, help='if resume')
    parser.add_argument('--resume_path', default='', type=str, help='if resume')
    parser.add_argument('--use_pretain', default='False', type=str, help='use pretained model')
    parser.add_argument('--pretrained_path', default='', type=str, help='if pretrained')
    parser.add_argument('--init_lr', default=2e-4, type=float)
    parser.add_argument('--save_path', default='./results_new/MTSSR3/', type=str)
    parser.add_argument('--log_dir', default='./results_new/MTSSR3/log', type=str)
    parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
    parser.add_argument("--trade_off", type=float, default=0, help="trade_off")
    parser.add_argument("--max_iter", type=float, default=125000, help="max_iter")  # 9240/15*200=1250 9240/8*200


    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    main(-1, args)
