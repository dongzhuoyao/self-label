'''Train CIFAR10 with PyTorch.
mostly from  https://github.com/zhirongw/lemniscate.pytorch/blob/master/cifar.py, AET
'''
from __future__ import print_function

import sys
import os
import argparse
import time

import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as tfs
from tensorboardX import SummaryWriter

from util import AverageMeter, setup_runtime, py_softmax
from cifar_utils import kNN, CIFAR10Instance, CIFAR100Instance

from pytorchgo.utils import logger
from pytorchgo.utils.pytorch_utils import model_summary, optimizer_summary
from tqdm import tqdm
import wandb
import pytorchgo_args
from wandb_wrapper import wandb_logging
import sklearn


def feature_return_switch(model, bool=True):
    """
    switch between network output or conv5features
        if True: changes switch s.t. forward pass returns post-conv5 features
        if False: changes switch s.t. forward will give full network output
    """
    if bool:
        model.headcount = 1
    else:
        model.headcount = args.hc
    model.return_features = bool
    if pytorchgo_args.get_args().method=="swav":
        model.return_features = True


def optimize_L_sk(PS,iter_num=10):
    PS = PS.cpu().numpy()
    N, K = PS.shape
    tt = time.time()
    PS = PS.T  # now it is K x N
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    PS **= args.lamb  # K x N
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    for _ in range(iter_num):
        r = inv_K / (PS @ c)  # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
    #logger.warning("sk error: {}".format(err))
    # inplace calculations.
    PS *= np.squeeze(c)
    PS = PS.T
    PS *= np.squeeze(r)
    PS = PS.T
    argmaxes = np.nanargmax(PS, 0)  # size N
    newL = torch.LongTensor(argmaxes)
    softlabels = newL.cuda()
    #logger.info('opt took {0:.2f}min'.format(((time.time() - tt) / 60.)))
    tim_sec = (time.time() - tt)
    PS = torch.from_numpy(PS.T).float().cuda()
    return  err,tim_sec, PS

def opt_sk(model, selflabels_in, epoch):
    if args.hc == 1:
        PS = np.zeros((len(trainloader.dataset), args.ncl))
    else:
        PS_pre = np.zeros((len(trainloader.dataset), knn_dim))
    for batch_idx, (data, _, _selected) in enumerate(trainloader):
        data = data.cuda()
        if args.hc == 1:
            p = nn.functional.softmax(model(data), 1)
            PS[_selected, :] = p.detach().cpu().numpy()
        else:
            p = model(data)
            PS_pre[_selected, :] = p.detach().cpu().numpy()
    if args.hc == 1:
         selflabels = optimize_L_sk(PS)
    else:
        _nmis = np.zeros(args.hc)
        nh = epoch % args.hc  # np.random.randint(args.hc)
        logger.info("computing head {}".format(nh))
        tl = getattr(model, "top_layer{}".format(nh))
        # do the forward pass:
        PS = (PS_pre @ tl.weight.cpu().numpy().T
                   + tl.bias.cpu().numpy())
        PS = py_softmax(PS, 1)
        selflabels_ = optimize_L_sk(PS)
        selflabels_in[nh] = selflabels_
        selflabels = selflabels_in
    return selflabels


datadir="cifar-10-batches-py"
bs = 128*5 #default of Sela is 128
epochs = 400
nopts = 400
hc = 1 #default of Sela is 10
arch ='alexnet'
ncl = 128 #default 128
cluter_num = 128 # special for SwaV
workers = 4
lr = 0.03
type =10
parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label for CIFAR10/100')

parser.add_argument('--device', default="0", type=str, help='cuda device')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--restart', action='store_true', help='restart opt')

# model
parser.add_argument('--arch', default=arch, type=str, help='architecture')
parser.add_argument('--ncl', default=ncl, type=int, help='number of clusters')
parser.add_argument('--hc', default=hc, type=int, help='number of heads')

# SK-optimization
parser.add_argument('--lamb', default=10.0, type=float, help='SK lambda parameter')
parser.add_argument('--nopts', default=nopts, type=int, help='number of SK opts')

# optimization
parser.add_argument('--lr', default=lr, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--epochs', default=epochs, type=int, help='number of epochs to train')
parser.add_argument('--batch-size', default=bs, type=int, metavar='BS', help='batch size')

# logging saving etc.
parser.add_argument('--datadir', default=datadir,type=str)
parser.add_argument('--type', default=type, type=int, help='cifar10 or 100')
parser.add_argument('--logger_option', default='d', type=str)
parser.add_argument("--wandb", type=bool, default=True)
parser.add_argument('--contrast_temp', default=0.07, type=float)
parser.add_argument('--cluter_num', default=cluter_num, type=float)
parser.add_argument('--method', default='swav', type=str)
parser.add_argument('--label', default='default', type=str)
parser.add_argument("--debug", action='store_true')

args = parser.parse_args()

pytorchgo_args.set_args(args)

#v1, no softmax in prototype initialization; fix 10 epochs.
customized_logger_dir = "train_log/v1_debug{debug}_swav_cifar{type}_pseudo{ncl}_{arch}_bs{bs}_hc{hc}-{nepochs}_nopt{nopts}_{label}".format(
        type=args.type,
    ncl=args.ncl,arch=args.arch,bs=args.batch_size, hc=args.hc, nepochs=args.epochs,nopts=args.nopts,debug=int(args.debug),label=args.label
    )

wandb.init(project="self-label", name=customized_logger_dir.replace("train_log/", ""))
logger.set_logger_dir(customized_logger_dir, args.logger_option)


setup_runtime(2, [args.device])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
knn_dim = 4096
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
logger.info('==> Preparing data..') #########################
transform_train = tfs.Compose([
    tfs.Resize(256),
    tfs.RandomResizedCrop(size=224, scale=(0.2, 1.)),
    tfs.ColorJitter(0.4, 0.4, 0.4, 0.4),
    tfs.RandomGrayscale(p=0.2),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = tfs.Compose([
        tfs.Resize(256),
        tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.type == 10:
    trainset = CIFAR10Instance(root=args.datadir, train=True, download=True,
                               transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)


    testset = CIFAR10Instance(root=args.datadir, train=False, download=True,
                              transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
else:

    trainset = CIFAR100Instance(root=args.datadir, train=True, download=True,
                               transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)


    testset = CIFAR100Instance(root=args.datadir, train=False, download=True,
                              transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


logger.info('==> Building model..') ##########################################
numc = [args.ncl] * args.hc
model = models.__dict__[args.arch](num_classes=numc,return_features=True)
knn_dim = 4096

N = len(trainloader.dataset)

def softmax(x,axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=axis,keepdims=True))
    return e_x / np.sum(e_x,axis=axis,keepdims=True)

# init selflabels randomly
if False:
    if args.hc == 1:
        selflabels = np.random.uniform(0, 1, (N, cluster_dim))
        selflabels = softmax(selflabels, axis=0)
        selflabels = torch.LongTensor(selflabels).cuda()
    else:
        selflabels = np.zeros((args.hc, N), dtype=np.int32)
        for nh in range(args.hc):
            for _i in range(N):
                selflabels[nh, _i] = _i % numc[nh]
            selflabels[nh] = np.random.permutation(selflabels[nh])
        selflabels = torch.LongTensor(selflabels).cuda()


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
# Model
if args.test_only or len(args.resume) > 0:
    # Load checkpoint.[
    logger.info('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    logger.info('loaded checkpoint at: ', checkpoint['epoch'])
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    if 'opt' in list(checkpoint.keys()):
        optimizer.load_state_dict(checkpoint['opt'])
    selflabels = checkpoint['L']
    selflabels = selflabels.to(device)
    include = [(qq / N >= start_epoch) for qq in optimize_times]
    optimize_times = (np.array(optimize_times)[include]).tolist()
    logger.info('We will optimize L at epochs:', [np.round(1.0 * t / N, 2) for t in optimize_times])
    model.to(device)
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.cuda()


model.to(device)
criterion = nn.CrossEntropyLoss()

if args.test_only:
    feature_return_switch(model, True)
    usepca = True
    acc = kNN(model, trainloader, testloader, K=[200, 50, 10, 5, 1], sigma=[0.1, 0.5], dim=knn_dim, use_pca=usepca)
    sys.exit(0)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if args.restart:
        if epoch == args.epochs//2:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=1e-6,
                                  nesterov=False)
    if args.epochs == 200:
        if epoch >= 80:
            lr = args.lr * (0.1 ** ((epoch - 80) // 40))  # i.e. 120, 160
            logger.info(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.epochs == 400:
        if epoch >= 160:
            lr = args.lr * (0.1 ** ((epoch - 160) // 80))  # i.e. 240,320
            logger.info(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.epochs == 800:
        if epoch >= 320:
            lr = args.lr * (0.1 ** ((epoch - 320) // 160))  # i.e. 480, 640
            logger.info(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.epochs == 1600:
        if epoch >= 640:
            lr = args.lr * (0.1 ** ((epoch - 640) // 320))
            logger.info(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# Training
def train(epoch):
    logger.info('Epoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    is_freeze_protoype=True

    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        niter = epoch * len(trainloader) + batch_idx
        pytorchgo_args.get_args().step += 1
        if args.debug and batch_idx >= 2: break
        if False:
            if niter * trainloader.batch_size >= optimize_times[-1]:
                with torch.no_grad():
                    _ = optimize_times.pop()
                    if args.hc >1:
                        feature_return_switch(model, True)
                    selflabels = opt_sk(model, selflabels, epoch)
                    if args.hc >1:
                        feature_return_switch(model, False)
        data_time.update(time.time() - end)
        inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        if epoch <= 10:#freeze update of prototype in epoch 0
            is_freeze_protoype = True
            model.prototype_N2K.requires_grad = False

        else:
            is_freeze_protoype = False
            model.prototype_N2K.requires_grad = True
            with torch.no_grad():
                _matrix = model.prototype_N2K
                normalize_dim = 0
                qn = torch.norm(_matrix, p=2,
                                dim=normalize_dim).detach()  # https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209/3
                model.prototype_N2K.data = _matrix.div(qn.unsqueeze(normalize_dim))
                # assert np.sum(model.prototype_N2K.detach().cpu().numpy()[:,0])<1.1


        scores = torch.mm(outputs, model.prototype_N2K)

        with torch.no_grad():
            err, tim_sec, q = optimize_L_sk(scores)
            #assert np.sum(q.detach().cpu().numpy()[:, 0]) < 1.1

        p = torch.softmax(scores/pytorchgo_args.get_args().contrast_temp, -1)

        loss = - torch.mean(q*torch.log(p))

        loss.backward()
        optimizer.step()



        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:
            logger.info('Epoch: [{}/{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}), best_knn_acc={best_knn_acc}, freeze={freeze}'.format(
                epoch,  args.epochs, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss,best_knn_acc=best_acc,freeze=is_freeze_protoype))
            wandb_logging(
                d=dict(loss1e4=loss.item()*1e4, group0_lr=optimizer.state_dict()['param_groups'][0]['lr'],sk_err=err,sk_time_sec=tim_sec),
                step=pytorchgo_args.get_args().step,
                use_wandb=pytorchgo_args.get_args().wandb,
                prefix="training epoch {}/{}: ".format(epoch, pytorchgo_args.get_args().epochs))
            #optimizer_summary(optimizer)

    cpu_prototype = model.prototype_N2K.detach().cpu().numpy()
    return cpu_prototype


optimizer_summary(optimizer)
model_summary(model)

pytorchgo_args.get_args().step = 0
for epoch in range(start_epoch, start_epoch + args.epochs):
    if args.debug and epoch >= 2: break
    prototype = train(epoch)
    feature_return_switch(model, True)
    logger.warning(logger.get_logger_dir())
    logger.warning("doing KNN evaluation.")
    acc = kNN(model, trainloader, testloader, K=10, sigma=0.1, dim=knn_dim)
    logger.warning("finish KNN evaluation.")
    feature_return_switch(model, False)
    if acc > best_acc:
        logger.info('get better result, saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'opt': optimizer.state_dict(),
            'prototype': prototype,
        }
        torch.save(state, os.path.join(logger.get_logger_dir(),'best_ckpt.t7'))
        best_acc = acc
    if epoch % 100 == 0:
        logger.info('Saving..')
        state = {
            'net': model.state_dict(),
            'opt': optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'prototype': prototype,
        }
        torch.save(state, os.path.join(logger.get_logger_dir(),'ep%s.t7'%(epoch)))
    if epoch % 50 == 0 and (not args.debug):
        feature_return_switch(model, True)
        acc4 = kNN(model, trainloader, testloader, K=[50, 10],
                  sigma=[0.1, 0.5], dim=knn_dim, use_pca=True)
        i = 0
        for num_nn in [50, 10]:
            for sig in [0.1, 0.5]:
                if args.debug:continue
                _str = 'knn{}-{}'.format(num_nn, sig)
                wandb_logging(
                    d={_str:acc4[i]},
                    step=pytorchgo_args.get_args().step,
                    use_wandb=pytorchgo_args.get_args().wandb,
                    prefix="")
                i += 1
        feature_return_switch(model, False)
    logger.info('best accuracy: {:.2f}'.format(best_acc * 100))
    wandb_logging(
        d=dict(knn_acc=acc,knn_best_acc=best_acc),
        step=pytorchgo_args.get_args().step,
        use_wandb=pytorchgo_args.get_args().wandb,
        prefix="")

checkpoint = torch.load(os.path.join(logger.get_logger_dir(),'best_ckpt.t7'))
model.load_state_dict(checkpoint['net'])
feature_return_switch(model, True)
acc = kNN(model, trainloader, testloader, K=10, sigma=0.1, dim=knn_dim, use_pca=True)
wandb.run.summary["final_test_acc"] = acc


