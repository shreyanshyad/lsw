from time import time
import math

from augmentations.cifar10policy import CIFAR10Policy
from config.arg_parser import init_parser

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config.dataset_config import *
from optimizer.sam import SAM
import torch.nn.functional as F
from models import MsViT


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        return self._compute_losses(x, target).mean()


best_acc1 = 0

epoch_loss = {}
epoch_loss['train'] = []
epoch_loss['val'] = []
epoch_acc = {}
epoch_acc['train'] = []
epoch_acc['val'] = []
epochs_list = []


def main():
    global best_acc1
    global epochs_list

    args = init_parser().parse_args()

    img_size = DATASETS[args.dataset]['img_size']
    num_classes = DATASETS[args.dataset]['num_classes']
    img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']
    arch = args.arch

    # l1,h8,d64,n1,s1,g1,p4,f7,a1_l2,h8,d128,n10,s0,g1,p2,f7,a1_l3,h8,d256,n1,s0,g1,p2,f7,a1

    model = MsViT(img_size=img_size, num_classes=num_classes,
                  arch=arch)

    criterion = LabelSmoothingCrossEntropy()

    epochs_list = [i+1 for i in range(args.epochs)]

    if torch.cuda.is_available():
        print("Using GPU")
        torch.cuda.set_device(0)
        model.cuda(0)
        criterion = criterion.cuda(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0)

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    augmentations = [CIFAR10Policy()]
    augmentations += [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ]

    augmentations = transforms.Compose(augmentations)
    train_dataset = datasets.__dict__[args.dataset.upper()](
        root=args.data, train=True, download=True, transform=augmentations)

    val_dataset = datasets.__dict__[args.dataset.upper()](
        root=args.data, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    print("Beginning training")
    print("Epochs: ", args.epochs)
    print("LR: ", args.lr)
    print("Layers: ", args.layers)
    print("Embedding dim: ", args.dim)
    print("Heads: ", args.heads)
    print("MLP: ", args.mlp)
    print("Dataset: ", args.dataset)
    print("Arch: ", args.arch)
    time_begin = time()
    for epoch in range(args.epochs):

        # adjust_learning_rate(optimizer, epoch, args)
        cls_train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = cls_validate(val_loader, model, criterion,
                            args, epoch=epoch, time_begin=time_begin)
        best_acc1 = max(acc1, best_acc1)
        scheduler.step()

    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'best top-1: {best_acc1:.2f}, '
          f'final top-1: {acc1:.2f}')
    torch.save(model.state_dict(), args.checkpoint_path)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(epochs_list, epoch_loss['train'], label='train')
    ax.plot(epochs_list, epoch_loss['val'], label='val')
    ax.set_title("Loss Epoch Graph")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.savefig('./epoch_loss.png')
    fig.show()
    # -----------------------------------
    fig, ax = plt.subplots()
    ax.plot(epochs_list, epoch_acc['train'], label='train')
    ax.plot(epochs_list, epoch_acc['val'], label='val')
    ax.set_title("Accuracy Epoch Graph")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.savefig('./epoch_acc.png')
    fig.show()


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if epoch >= 20 and epoch < 30:
        lr = 0.001
    elif epoch >= 30 and epoch < 35:
        lr = 0.0006
    elif epoch >= 35:
        lr = 0.0003
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return None

    tot_epochs = args.epochs
    LR = args.lr
    stepsize = 2
    k_min = 0.1
    k_max = 0.08

    N_MIN = LR*0.9
    N_MAX = LR

    n_min = N_MIN
    n_max = N_MAX

    lr = LR

    for epoch in range(1, tot_epochs+1):
        warmup = args.warmup
        if epoch <= warmup:
            lr = lr
        else:
            ep = epoch-warmup
            n_max = N_MAX*math.exp(-ep*k_max)
            n_min = N_MIN*math.exp(-ep*k_min)
            cycle = 1+ep//(2*stepsize)
            x = abs(ep/stepsize-2*cycle+1)
            lr = n_min+(n_max-n_min)*max(0, 1-x)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cls_train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    for i, (images, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

        def closure():
            loss = criterion(model(images), target)
            loss.backward()
            return loss

        output = model(images)

        loss = criterion(output, target)

        acc1 = accuracy(output, target)
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        loss.backward()
        optimizer.step(closure)
        optimizer.zero_grad()

        # if args.clip_grad_norm > 0:
        #     nn.utils.clip_grad_norm_(
        #         model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            print(
                f'[Epoch {epoch + 1}][Train][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

    epoch_loss['train'].append(loss_val/n)
    epoch_acc['train'].append(acc1_val/n)


def cls_validate(val_loader, model, criterion, args, epoch=None, time_begin=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(0, non_blocking=True)
                target = target.cuda(0, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                print(
                    f'[Epoch {epoch + 1}][Eval][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    epoch_loss['val'].append(avg_loss)
    epoch_acc['val'].append(avg_acc1)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(
        f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc1:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_acc1


if __name__ == '__main__':
    main()
