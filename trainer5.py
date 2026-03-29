import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from itertools import cycle
import torchvision
import torch.distributed as dist
from torch.optim import lr_scheduler
import PIL.Image as Image
from utils import *
from torch.autograd import Variable
from adamp import AdamP
from torchvision.models import vgg16
import torch.optim as optim
from losses.self_test_hyper import Loss_hyper_loss, Loss_hyper2
from losses.train_losses2 import PerpetualLoss, Loss_train_RMSE, Loss_PSNR2, Loss_RMSE, Loss_valid, LossTrainCSS2, initialize_logger, record_loss3, Constrast_loss_3
# import pyiqa
import os
import time
import copy
import shutil


class Trainer:
    def __init__(self, model, tmodel, cmodel, args, supervised_loader, unsupervised_loader, val_loader, test_loader, iter_per_epoch, writer):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.iter_per_epoch = iter_per_epoch
        self.writer = writer
        self.model = model
        self.tmodel = tmodel
        self.cmodel = cmodel
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.num_epochs
        self.save_period = 200
        self.self_rgb = LossTrainCSS2()
        self.loss_unsup = Loss_train_RMSE().cuda()
        self.loss_sup_hyper = Loss_hyper2().cuda()
        self.cross_hyper = Loss_hyper2().cuda()
        self.loss_str = Loss_train_RMSE().cuda()
        self.loss_hyper_sup = Loss_hyper_loss().cuda()
        
        self.consistency = 0.2
        self.consistency_rampup = 100.0
        self.criterion_valid_mrae = Loss_valid() #mrae
        self.criterion_valid_psnr = Loss_PSNR2()
        self.criterion_valid_rmse = Loss_RMSE()

        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        self.loss_per = PerpetualLoss(vgg_model).cuda()
        self.constrast = Constrast_loss_3(vgg_model).cuda()
        self.curiter = 0
        self.model.cuda()
        self.tmodel.cuda()
        self.cmodel.cuda()

        self.optimizer_s = optim.Adam(self.model.parameters(), lr=self.args.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.996):
        # exponential moving average(EMA)
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def predict_with_out_grad(self, image):
        with torch.no_grad():
            predict_target_ul = self.tmodel(image)

        return predict_target_ul

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False

    def freeze_cmodel_parameters(self):
        for p in self.cmodel.parameters():
            p.requires_grad = False
    
    def train(self):
        loss_csv = open(os.path.join(self.args.save_path, 'loss.csv'), 'a+')
        log_dir = os.path.join(self.args.save_path, 'train.log')
        logger = initialize_logger(log_dir)
        record_val_loss = 1000
        record_test_loss = 1000
        iteration = 0
        best_epoch = 0
        model_list = []
        shutil.copy('/home/lyh/workspace/ConvMLP_2020/trainer7.py', self.args.save_path)
        shutil.copy('/home/lyh/workspace/ConvMLP_2020/train3.py', self.args.save_path)
        shutil.copy('/home/lyh/workspace/ConvMLP_2020/losses/train_losses2.py', self.args.save_path)


        self.freeze_teachers_parameters()
        self.freeze_cmodel_parameters()
        if self.start_epoch == 1:
            initialize_weights(self.model)
        else:
            iteration = 17864
            checkpoint = torch.load(self.args.resume_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.tmodel.load_state_dict(checkpoint['state_dict_teacher'])
            model_list = checkpoint['bank']

        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.time()

            loss_ave, loss_sup, sup_hyper_loss,loss_unsup,losses_rgb, losses_cons, iteration,lr = self._train_epoch(epoch, iteration, best_epoch, model_list)
            val_rmae, val_rmse, val_psnr = self._valid_epoch(epoch)        
            test_rmae, test_rmse, test_psnr = self._test_epoch(epoch)    

            if test_rmae < record_test_loss:
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict()}
                record_test_loss = test_rmae
                ckpt_name = str(self.args.save_path) + 'model_e{}.pth'.format(str(epoch))
                print("Saving a checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)

            if torch.abs(val_rmae - record_val_loss) < 0.00001 or val_rmae < record_val_loss:
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'state_dict_teacher': self.tmodel.state_dict(),
                         'bank': model_list,
                         'optimizer_dict': self.optimizer_s.state_dict()}
                # 更新最新epoch
                best_epoch = epoch
                record_val_loss = val_rmae
                
                ckpt_name = str(self.args.save_path) + 'model_e{}.pth'.format(str(epoch))
                print("Saving a checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)
            # print loss

            if epoch == best_epoch:
                if len(model_list) == 2:
                    model_list.pop(0)
                    model_list.append(self.tmodel.state_dict())
                else:
                    model_list.append(self.tmodel.state_dict())
            else:
                print('Go on searching!')

            end_time = time.time()
            epoch_time = end_time - start_time

            print('Epoch[%d] Iter[%d] Time:%.9f, main_loss: %.6f, sup loss: %.6f, sup hyper loss: %.6f, usup loss: %.6f, test rmae: %.6f, test rmse: %.6f, test psnr: %.6f, val_rmae: %.6f, val_rmse: %.6f, val psnr: %.6f' % (
                epoch, iteration, epoch_time,  loss_ave, loss_sup, sup_hyper_loss, loss_unsup, test_rmae, test_rmse, test_psnr, val_rmae, val_rmse, val_psnr))
            record_loss3(loss_csv, epoch, iteration, epoch_time,  loss_ave, loss_sup, sup_hyper_loss, loss_unsup, losses_rgb, losses_cons, test_rmae, test_rmse, test_psnr, val_rmae, val_rmse, val_psnr, lr)
            logger.info('Epoch[%d] Iter[%d] Time:%.9f, main_loss: %.6f, sup loss: %.6f, sup hyper loss: %.6f, usup loss: %.6f, losses_rgb %.6f, losses_cons %.6f, test rmae: %.6f, test rmse: %.6f, test psnr: %.6f, val_rmae: %.6f, val_rmse: %.6f, val psnr: %.6f, lr: %.6f' % (
                epoch, iteration, epoch_time,  loss_ave, loss_sup, sup_hyper_loss, loss_unsup,losses_rgb, losses_cons, test_rmae, test_rmse, test_psnr, val_rmae, val_rmse, val_psnr, lr))

            

    def _train_epoch(self, epoch, iteration, best_epoch, model_list):
        sup_loss = AverageMeter()
        sup_hyper_loss = AverageMeter()
        unsup_loss = AverageMeter()
        losses_rgb = AverageMeter()
        losses_cons = AverageMeter()
        all_loss = AverageMeter()
        self.model.train()
        self.freeze_teachers_parameters()
        train_loader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
        tbar = range(len(self.unsupervised_loader))
        # 进度条展示图
        tbar = tqdm(tbar, ncols=130, leave=True)
        for i in tbar:
            (pair_img, label), (unpair_img , no_label, bank) = next(train_loader)
            pair_img = Variable(pair_img).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            unpair_img = Variable(unpair_img).cuda(non_blocking=True)
            Bank = Variable(bank).cuda(non_blocking=True)
            # 计算学习率部分
            
            lr = self.poly_lr_scheduler(self.optimizer_s, self.args.init_lr, iteration, self.args.max_iter,self.args.decay_power)
            iteration = iteration + 1
            predict_target_u = self.predict_with_out_grad(unpair_img)
            # student output
            outputs_l = self.model(pair_img)
            outputs_ul = self.model(unpair_img)

            if epoch < 2:
                n_sample = Bank.detach() # 最开始这部分是0值
                p_sample = predict_target_u.detach() # 最开始

            elif epoch >= 2 and epoch == best_epoch + 1:
                self.cmodel.load_state_dict(model_list[-1])
                n_sample = self.cmodel(unpair_img)

                p_sample = predict_target_u.detach()
            else :
                # self.cmodel.load_state_dict(model_list[-1])
                # p_sample = self.cmodel(unpair_img)
                p_sample = predict_target_u.detach() 
                if len(model_list) >= 2:
                    self.cmodel.load_state_dict(model_list[0])
                else:
                    self.cmodel.load_state_dict(model_list[-1])
                n_sample = self.cmodel(unpair_img)

            

            # 约束函数
            structure_loss = self.loss_str(outputs_l, label)
            hyper_loss = self.loss_per(outputs_l,label)
            loss_sup = structure_loss + 0.1 * hyper_loss
            
            loss_unsu = self.loss_unsup(outputs_ul, p_sample) 
            loss_rgb = self.self_rgb(outputs_ul,unpair_img)
            loss_cons = self.constrast(p_sample, n_sample, outputs_ul)

            loss_U = (loss_unsu + 0.1*loss_rgb) +1e-6 * loss_cons


            total_loss =loss_sup + 0.1*loss_U

            sup_loss.update(loss_sup.data)
            sup_hyper_loss.update(hyper_loss.data)
            unsup_loss.update(loss_unsu.data)
     
            losses_rgb.update(loss_rgb.data)
            all_loss.update(total_loss.data)
            losses_cons.update(loss_cons.data)

            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()

            tbar.set_description('Train-Student Epoch {} | Ls {:.4f} Lhyper {:.4f} Lu {:.4f} Lrgb {:.4f} Lcons {:.4f}|'
                                 .format(epoch, sup_loss.avg, sup_hyper_loss.avg, unsup_loss.avg, losses_rgb.avg, losses_cons.avg))

            del pair_img, label, unpair_img, no_label
            with torch.no_grad():
                self.update_teachers(teacher=self.tmodel, itera=self.curiter)
                self.curiter = self.curiter + 1

        self.writer.add_scalar('Train_loss', all_loss.avg, global_step=epoch)
        self.writer.add_scalar('sup_loss', sup_loss.avg, global_step=epoch)
        self.writer.add_scalar('sup_hyper_loss', sup_hyper_loss.avg, global_step=epoch)
        self.writer.add_scalar('unsup_loss', unsup_loss.avg, global_step=epoch)
        self.writer.add_scalar('rgb_loss', losses_rgb.avg, global_step=epoch)
        self.writer.add_scalar('cons_loss', losses_cons.avg, global_step=epoch)
        # self.lr_scheduler_s.step(epoch=epoch - 1)
        return all_loss.avg, sup_loss.avg, sup_hyper_loss.avg, unsup_loss.avg, losses_rgb.avg, losses_cons.avg, iteration, lr

    def _valid_epoch(self, epoch):
        self.model.eval()

        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for i, (val_data, val_label) in enumerate(tbar):
                val_data = Variable(val_data).cuda()
                val_label = Variable(val_label).cuda()

                val_output= self.model(val_data)
                # print('output:{}'.format(val_output.max()))
                loss_mrae = self.criterion_valid_mrae(val_output, val_label)
                loss_rmse = self.criterion_valid_rmse(val_output, val_label)
                loss_psnr = self.criterion_valid_psnr(val_output, val_label)

                losses_mrae.update(loss_mrae.data)
                losses_rmse.update(loss_rmse.data)
                losses_psnr.update(loss_psnr.data)

            self.writer.add_scalar('Val_rmse', losses_mrae.avg, global_step=epoch)
            self.writer.add_scalar('Val_mrae', losses_rmse.avg, global_step=epoch)
            self.writer.add_scalar('Val_psnr', losses_psnr.avg, global_step=epoch)
            del val_output, val_label, val_data
            return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg
    
    def _test_epoch(self, epoch):
        self.model.eval()
        self.tmodel.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        tbar = tqdm(self.test_loader, ncols=130)
        with torch.no_grad():
            for i, (test_data, test_label) in enumerate(tbar):
                test_data = Variable(test_data).cuda()
                test_label = Variable(test_label).cuda()

                # forward
                # print('label:{}'.format(val_label.max()))
                test_output= self.model(test_data)
                # print('output:{}'.format(val_output.max()))
                loss_mrae = self.criterion_valid_mrae(test_output, test_label)
                loss_rmse = self.criterion_valid_rmse(test_output, test_label)
                loss_psnr = self.criterion_valid_psnr(test_output, test_label)

                losses_mrae.update(loss_mrae.data)
                losses_rmse.update(loss_rmse.data)
                losses_psnr.update(loss_psnr.data)

            self.writer.add_scalar('Test_mrae', losses_mrae.avg, global_step=epoch)

            self.writer.add_scalar('Test_rmse', losses_rmse.avg, global_step=epoch)
            self.writer.add_scalar('Test_psnr', losses_psnr.avg, global_step=epoch)
            del test_label, test_output, test_data
            return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:1' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        
    def poly_lr_scheduler(self,optimizer, init_lr, iteraion, max_iter, power):
        """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

        """
        if iteraion % 1.0 or iteraion > max_iter:
            return optimizer

        lr = init_lr*(1 - iteraion/max_iter)**power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print('lr1:{}'.format(lr))

        return lr
