import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from models.mnist_net import mnist_net

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--data_dir', default='./data_mnist', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--attack', default='fgsm_uni', type=str, choices=['fgsm_uni', 'fgsm_half_bound', 'fgsm_bound'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--alpha_ratio', default=0.5, type=float)
    parser.add_argument('--clamp', action='store_true')
    parser.add_argument('--pgd_attack_iters', default=50, type=int)
    parser.add_argument('--pgd_alpha_value', default=1e-2, type=float)
    parser.add_argument('--pgd_restarts', default=10, type=int)
    parser.add_argument('--lr-max', default=5e-3, type=float)
    parser.add_argument('--lr_type', default='cyclic')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()

def clamp_mnist(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_fgsm_mnist(attack, clamp, model, X, y, epsilon, alpha_ratio):
    delta = torch.zeros_like(X).cuda()
    if attack == 'fgsm_uni':
        delta.uniform_(-epsilon, epsilon)
    elif attack == 'fgsm_half_bound':
        delta = delta.normal_(0, 1).sign() * epsilon / 2
    elif attack == 'fgsm_bound':
        delta = delta.normal_(0, 1).sign() * epsilon
    delta.requires_grad = True
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    if clamp:
        delta.data = torch.clamp(delta + alpha_ratio * epsilon * torch.sign(grad), -epsilon, epsilon)
    else:
        delta.data = delta + alpha_ratio * epsilon * torch.sign(grad)
    delta.data = clamp_mnist(delta.data, 0 - X, 1 - X)
    return delta.detach()


def attack_pgd_mnist(model, X, y, epsilon, alpha_value, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        delta.data = clamp_mnist(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha_value * torch.sign(grad), -epsilon, epsilon)
            d = clamp_mnist(d, 0-X, 1-X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def eval(args, model, test_loader, criterion, finaleval=False):
    model.eval()
    test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc, test_pgd_delta_norm, test_n = 0, 0, 0, 0, 0, 0
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        clean_outputs = model(X)
        clean_loss = criterion(clean_outputs, y)

        if finaleval:
            pgd_delta = attack_pgd_mnist(model, X, y, args.epsilon, args.pgd_alpha_value, 50, 10).detach()
        else:
            pgd_delta = attack_pgd_mnist(model, X, y, args.epsilon, args.pgd_alpha_value, args.pgd_attack_iters,
                                     args.pgd_restarts).detach()
        pgd_outputs = model(X + pgd_delta)
        pgd_loss = criterion(pgd_outputs, y)
        pgd_delta_norm = pgd_delta.view(pgd_delta.shape[0], -1).norm(dim=1)

        test_clean_loss += clean_loss.item() * y.size(0)
        test_clean_acc += (clean_outputs.max(1)[1] == y).sum().item()
        test_pgd_loss += pgd_loss.item() * y.size(0)
        test_pgd_acc += (pgd_outputs.max(1)[1] == y).sum().item()
        test_pgd_delta_norm += pgd_delta_norm.sum().item()
        test_n += y.size(0)
    return test_clean_loss / test_n, 100. * test_clean_acc / test_n, test_pgd_loss / test_n, 100. * test_pgd_acc / test_n, test_pgd_delta_norm / test_n


def tb_writer(writer, epoch, lr, train_clean_loss, train_clean_acc, train_fgsm_loss, train_fgsm_acc, train_fgsm_delta_norm,
              test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc, test_pgd_delta_norm):
    writer.add_scalars('loss',
                       {'train_clean': train_clean_loss, 'train_fgsm': train_fgsm_loss,
                        'test_clean': test_clean_loss, 'test_pgd': test_pgd_loss}, epoch)
    writer.add_scalars('accuracy',
                       {'train_clean': train_clean_acc, 'train_fgsm': train_fgsm_acc,
                        'test_clean': test_clean_acc, 'test_pgd': test_pgd_acc}, epoch)
    writer.add_scalar('learning rate', lr, epoch)
    writer.add_scalars('delta_norm', {'train_fgsm': train_fgsm_delta_norm, 'test_pgd': test_pgd_delta_norm}, epoch)


def main():
    args = get_args()

    OUTPUT_DIR = './output'
    LOG_DIR = './output/log'
    TENSORBOARD_DIR = './output/tensorboard'
    CHECKPOINT_DIR = '../../../../scratch/pekang/checkpoint/'
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.exists(TENSORBOARD_DIR):
        os.mkdir(TENSORBOARD_DIR)
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, args.exp_name))

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        filename=os.path.join(LOG_DIR, args.exp_name))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    mnist_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)

    model = mnist_net().cuda()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    if args.lr_type == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat':
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError('Unknown lr_type')


    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train clean Loss \t Train clean Acc \t Train fgsm loss \t Train fgsm acc \t Test Standard Loss \t Test Standard '
        'Acc \t Test Attack Loss \t Test Attack Acc \t Train fgsm norm \t Test pgd norm')
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_clean_loss, train_clean_acc, train_fgsm_loss, train_fgsm_acc, train_fgsm_delta_norm, train_n = 0, 0, 0, 0, 0, 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            clean_outputs = model(X)
            clean_loss = criterion(clean_outputs, y)

            delta_fgsm = attack_fgsm_mnist(args.attack, args.clamp, model, X, y, args.epsilon, args.alpha_ratio)
            output_fgsm = model(X + delta_fgsm)
            loss_fgsm = criterion(output_fgsm, y)
            delta_fgsm_norm = delta_fgsm.view(delta_fgsm.shape[0], -1).norm(dim=1)
            opt.zero_grad()
            loss_fgsm.backward()
            opt.step()

            train_clean_loss += clean_loss.item() * y.size(0)
            train_clean_acc += (clean_outputs.max(1)[1] == y).sum().item()
            train_fgsm_loss += loss_fgsm.item() * y.size(0)
            train_fgsm_acc += (output_fgsm.max(1)[1] == y).sum().item()
            train_fgsm_delta_norm += delta_fgsm_norm.sum().item()
            train_n += y.size(0)

        train_time = time.time()
        test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc, test_pgd_delta_norm = eval(args, model, test_loader, criterion)
        test_time = time.time()

        logger.info(
            '%d %.1f %.1f %.4f  %.4f %.2f  %.4f %.2f  %.4f %.2f  %.4f %.2f  %.2f %.2f',
            epoch+1, train_time-start_time, test_time-train_time, lr,
            train_clean_loss / train_n, 100. * train_clean_acc / train_n, train_fgsm_loss / train_n, 100. * train_fgsm_acc / train_n,
            test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc, train_fgsm_delta_norm / train_n, test_pgd_delta_norm)
        tb_writer(writer, epoch+1, lr, train_clean_loss / train_n, 100. * train_clean_acc / train_n, train_fgsm_loss / train_n, 100. * train_fgsm_acc / train_n, train_fgsm_delta_norm / train_n,
            test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc, test_pgd_delta_norm)

    final_clean_loss, final_clean_acc, final_pgd_loss, final_pgd_acc, final_pgd_delta_norm = eval(args, model, test_loader, criterion, finaleval=True)
    logger.info(' %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f', final_clean_loss, final_clean_acc, final_pgd_loss, final_pgd_acc)


if __name__ == "__main__":
    main()