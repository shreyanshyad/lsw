import argparse


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CIFAR quick training script')
    # Data args
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset', default='./cifar-10-dataset')
    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['cifar10', 'cifar100'],
                        default='cifar10')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                        help='log frequency (by iteration)')
    parser.add_argument('--checkpoint-path',
                        type=str,
                        default='checkpoint.pth',
                        help='path to checkpoint (default: checkpoint.pth)')
    # Optimization hyperparams
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=5, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)', dest='batch_size')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--layers', default=7, type=int,
                        help='layers (default 7)')
    parser.add_argument('--dim', default=256, type=int,
                        help='embedding dim (default 256)')
    parser.add_argument('--heads', default=4, type=int,
                        help='heads num (4)')
    parser.add_argument('--mlp', default=2, type=float,
                        help='mlp ratio (2)')
    return parser
