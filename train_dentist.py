import os
import sys
import platform
import socket
import argparse
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

import mxnet as mx
from mxnet import gluon, autograd, init
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs, LRScheduler, LRSequential

from data import IceContrast
from model import ASKCResNetFPN, ASKCResUNet
from metric import SigmoidMetric, SamplewiseSigmoidMetric
from loss import SoftIoULoss
from utils import summary


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon Segmentation')
    parser.add_argument('--host', type=str, default='xxx',
                        help='xxx is a place holder')
    parser.add_argument('--model', type=str, default='ResFPN',
                        help='model name: ResNetFPN, ResUNet')
    parser.add_argument('--fuse-mode', type=str, default='AsymBi',
                        help='DirectAdd, Concat, SK, BiLocal, BiGlobal, AsymBi, '
                             'TopDownGlobal, TopDownLocal')
    parser.add_argument('--tiny', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--blocks', type=int, default=3,
                        help='block num in each stage')
    parser.add_argument('--channel-times', type=int, default=1,
                        help='times of channel width')
    parser.add_argument('--dataset', type=str, default='DENTIST',
                        help='dataset name: DENTIST, Iceberg, StopSign')
    parser.add_argument('--workers', type=int, default=48,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='iou-thresh')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='trainval',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', type=str, default='test',
                        help='dataset val split (default: val)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--optimizer', type=str, default='adagrad',
                        help='sgd, adam, adagrad')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--gamma', type=int, default=2,
                        help='gamma for Focal Soft IoU Loss')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    parser.add_argument('--score-thresh', type=float, default=0.5,
                        help='score-thresh')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--colab', action='store_true', default=
                        False, help='whether using colab')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='directory of saved models')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
    parser.add_argument('--metric', type=str, default='mAP',
                        help='F1, IoU, mAP')
    parser.add_argument('--logging-file', type=str, default='train.log',
                        help='name of training log file')
    parser.add_argument('--summary', action='store_true',
                        help='print parameters')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda or (len(mx.test_utils.list_gpus()) == 0):
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        args.ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        print('Number of GPUs:', len(args.ctx))

    # logging and checkpoint saving
    if args.save_dir is None:
        args.save_dir = "runs/%s/%s/" % (args.dataset, args.model)
    makedirs(args.save_dir)

    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': len(args.ctx)} if args.syncbn else {}
    print(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args

        filehandler = logging.FileHandler(args.logging_file)
        streamhandler = logging.StreamHandler()

        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(filehandler)
        self.logger.addHandler(streamhandler)

        self.logger.info(args)


        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        ])

        ################################# dataset and dataloader #################################
        if platform.system() == "Darwin":
            data_root = os.path.join('~', 'Nutstore Files', 'Dataset')  # Mac
        elif platform.system() == "Linux":
            data_root = os.path.join('~', 'datasets')  # Laplace or HPC
            if args.colab:
                data_root = '/content/datasets'  # Colab
        else:
            raise ValueError('Notice Dataset Path')

        data_kwargs = {'base_size': args.base_size, 'transform': input_transform,
                       'crop_size': args.crop_size, 'root': data_root,
                       'base_dir': args.dataset}
        trainset = IceContrast(split=args.train_split, mode='train',   **data_kwargs)
        valset = IceContrast(split=args.val_split, mode='testval', **data_kwargs)
        self.train_data = gluon.data.DataLoader(trainset, args.batch_size, shuffle=True,
                                                last_batch='rollover', num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
                                               last_batch='rollover', num_workers=args.workers)

        layers = [args.blocks] * 3
        channels = [x * args.channel_times for x in [8, 16, 32, 64]]
        if args.model == 'ResNetFPN':
            model = ASKCResNetFPN(layers=layers, channels=channels, fuse_mode=args.fuse_mode,
                                  tiny=args.tiny, classes=trainset.NUM_CLASS)
        elif args.model == 'ResUNet':
            model = ASKCResUNet(layers=layers, channels=channels, fuse_mode=args.fuse_mode,
                                tiny=args.tiny, classes=trainset.NUM_CLASS)
        print("layers: ", layers)
        print("channels: ", channels)
        print("fuse_mode: ", args.fuse_mode)
        print("tiny: ", args.tiny)
        print("classes: ", trainset.NUM_CLASS)

        if args.host == 'xxx':
            self.host_name = socket.gethostname()  # automatic
        else:
            self.host_name = args.host             # Puma needs to be specified
        self.save_prefix = '_'.join([args.model, args.fuse_mode, args.dataset, self.host_name,
                            'GPU', args.gpus])

        model.cast(args.dtype)
        # self.logger.info(model)

        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                model.load_parameters(args.resume, ctx=args.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        else:
            model.initialize(init=init.MSRAPrelu(), ctx=args.ctx, force_reinit=True)
            print("Model Initializing")
            print("args.ctx: ", args.ctx)

        self.net = model
        if args.summary:
            summary(self.net, mx.nd.zeros((1, 3, args.crop_size, args.crop_size), ctx=args.ctx[0]))
            sys.exit()

        # create criterion
        self.criterion = SoftIoULoss()

        # optimizer and lr scheduling
        self.lr_scheduler = LRSequential([
                LRScheduler('linear', base_lr=0, target_lr=args.lr,
                            nepochs=args.warmup_epochs, iters_per_epoch=len(self.train_data)),
                LRScheduler(mode='poly', base_lr=args.lr,
                            nepochs=args.epochs-args.warmup_epochs,
                            iters_per_epoch=len(self.train_data),
                            power=0.9)
            ])
        kv = mx.kv.create(args.kvstore)

        if args.optimizer == 'sgd':
            optimizer_params = {'lr_scheduler': self.lr_scheduler,
                                'wd': args.weight_decay,
                                'momentum': args.momentum,
                                'learning_rate': args.lr}
        elif args.optimizer == 'adam':
            optimizer_params = {'lr_scheduler': self.lr_scheduler,
                                'wd': args.weight_decay,
                                'learning_rate': args.lr}
        elif args.optimizer == 'adagrad':
            optimizer_params = {
                'wd': args.weight_decay,
                'learning_rate': args.lr
            }
        else:
            raise ValueError('Unsupported optimizer {} used'.format(args.optimizer))

        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        self.optimizer = gluon.Trainer(self.net.collect_params(), args.optimizer,
                                       optimizer_params, kvstore=kv)

        ################################# evaluation metrics #################################

        self.iou_metric = SigmoidMetric(1)
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=self.args.score_thresh)
        self.best_iou = 0
        self.best_nIoU = 0
        self.is_best = False

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        for i, batch in enumerate(tbar):
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.args.ctx, batch_axis=0)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=self.args.ctx, batch_axis=0)
            losses = []
            with autograd.record(True):
                for x, y in zip(data, labels):
                    pred = self.net(x)
                    loss = self.criterion(pred, y.astype(self.args.dtype, copy=False))
                    losses.append(loss)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)
            for loss in losses:
                train_loss += np.mean(loss.asnumpy()) / len(losses)
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, train_loss/(i+1)))
            if i != 0 and i % self.args.log_interval == 0:
                self.logger.info('Epoch %d iteration %04d/%04d: training loss %.3f' % \
                    (epoch, i, len(self.train_data), train_loss/(i+1)))
            mx.nd.waitall()

    def validation(self, epoch):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        tbar = tqdm(self.eval_data)
        for i, batch in enumerate(tbar):
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.args.ctx, batch_axis=0)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=self.args.ctx, batch_axis=0)
            preds = []
            for x, y in zip(data, labels):
                pred = self.net(x)
                preds.append(pred)
            self.iou_metric.update(preds, labels)
            self.nIoU_metric.update(preds, labels)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            tbar.set_description('Epoch %d, IoU: %.4f, nIoU: %.4f' % (epoch, IoU, nIoU))

        if IoU > self.best_iou:
            self.best_iou = IoU
            self.net.save_parameters('%s/%.4f-%s-%s-%d-best.params'%
                                     (self.args.save_dir, IoU, self.save_prefix, 'IoU', epoch))
            with open(self.save_prefix + '_best_IoU.log', 'a') as f:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epoch, IoU))

        if nIoU > self.best_nIoU:
            self.best_nIoU = nIoU
            self.net.save_parameters('%s/%.4f-%s-%s-%d-best.params'%
                                     (self.args.save_dir, nIoU, self.save_prefix, 'nIoU', epoch))
            with open(self.save_prefix + '_best_nIoU.log', 'a') as f:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epoch, nIoU))

        if epoch >= args.epochs - 1:
            with open(self.save_prefix + '_best_IoU.log', 'a') as f:
                f.write('Finished\n')
            with open(self.save_prefix + '_best_nIoU.log', 'a') as f:
                f.write('Finished\n')
            print("best_iou: ", self.best_iou)
            print("best_nIoU: ", self.best_nIoU)

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epochs:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)
