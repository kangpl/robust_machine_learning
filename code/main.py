"""Train CIFAR10 with PyTorch."""

import argparse
import logging
import os
import time

import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18
from utils.util import *
from deepfool import deepfool
from cure import cure
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset_path', default='./data', help='path of the dataset')

    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', default='multistep', choices=['multistep', 'constant'])
    parser.add_argument('--lr_change_epoch', nargs='+', default=[100, 150], type=int)
    parser.add_argument('--batch_size', '-b', default=128, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)

    parser.add_argument('--cure', action='store_true', help='use curvature regularization during training')
    parser.add_argument('--cure_lambda', default=4, type=int)
    parser.add_argument('--cure_h', default=1.5, type=float)
    parser.add_argument('--attack_during_train', default='fgsm', type=str, choices=['pgd', 'fgsm', 'none','both'])
    parser.add_argument('--train_epsilon', default=8, type=int)
    parser.add_argument('--train_fgsm_alpha', default=10, type=float)
    parser.add_argument('--train_pgd_alpha', default=2, type=float)
    parser.add_argument('--train_pgd_attack_iters', default=10, type=int)
    parser.add_argument('--train_pgd_restarts', default=1, type=int)
    parser.add_argument('--attack_during_test', default='pgd', type=str, choices=['pgd', 'deepfool', 'none'])
    parser.add_argument('--test_epsilon', default=8, type=int)
    parser.add_argument('--test_pgd_alpha', default=2, type=float)
    parser.add_argument('--test_pgd_attack_iters', default=10, type=int)
    parser.add_argument('--test_pgd_restarts', default=1, type=int)
    parser.add_argument('--deepfool_classes_num', default=2, type=int)
    parser.add_argument('--deepfool_max_iter', default=50, type=int)

    parser.add_argument('--finetune', action='store_true', help='finetune the pre-trained model with adversarial '
                                                                'samples or regularization')
    parser.add_argument('--resumed_model_name', default='standard_cifar.pth', help='the file name of resumed model')
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()


# Training
def train(args, model, trainloader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    train_norm = []
    train_total = 0
    df_loop = []
    df_perturbation_norm = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        if args.cure:
            outputs = model(inputs)
            if epoch < 5:
                loss = criterion(outputs, targets) + args.cure_lambda * cure(model, inputs, targets, h=args.cure_h / 5 * (epoch + 1), device=args.device)
            else:
                loss = criterion(outputs, targets) + args.cure_lambda * cure(model, inputs, targets, h=args.cure_h, device=args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            if args.attack_during_train == 'pgd':
                delta = attack_pgd(model, inputs, targets, args.train_epsilon, args.train_pgd_alpha,
                                   args.train_pgd_attack_iters, args.train_pgd_restarts, args.device)
            elif args.attack_during_train == 'fgsm':
                delta = attack_pgd(model, inputs, targets, args.train_epsilon, args.train_fgsm_alpha, 1, 1, args.device)
            elif args.attack_during_train == 'none':
                delta = torch.zeros_like(inputs)
            delta = delta.detach()

            outputs = model(clamp(inputs + delta, lower_limit, upper_limit))
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        inputs_grad = get_input_grad(model, inputs, targets, delta_init='none', backprop=False, device=args.device)
        # inputs_grad = get_input_grad_v2(model, inputs, targets)
        norm = inputs_grad.view(inputs_grad.shape[0], -1).norm(dim=1)

        # deepfool attack
        if batch_idx % 10 == 0:
            pert_inputs, loop, perturbation = deepfool(model, inputs, targets, num_classes=args.deepfool_classes_num,
                                                       max_iter=args.deepfool_max_iter, device=args.device)
            df_loop.append(loop.cpu().numpy())
            df_perturbation_norm.append(perturbation.cpu().numpy())

        train_loss += loss.item() * targets.size(0)
        train_correct += (outputs.max(dim=1)[1] == targets).sum().item()
        train_norm.append(norm.cpu().numpy())
        train_total += targets.size(0)
    all_norm = np.concatenate(train_norm)
    df_loop = np.concatenate(df_loop)
    df_perturbation_norm = np.concatenate(df_perturbation_norm)
    return train_loss / train_total, 100. * train_correct / train_total, all_norm, df_loop, df_perturbation_norm


def test(args, model, testloader, criterion):
    model.eval()
    test_standard_loss = 0
    test_standard_correct = 0
    test_attack_loss = 0
    test_attack_correct = 0
    test_norm = []
    df_loop = []
    df_perturbation_norm = []
    test_total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        if args.attack_during_test == 'pgd':
            delta = attack_pgd(model, inputs, targets, args.test_epsilon, args.test_pgd_alpha,
                               args.test_pgd_attack_iters,
                               args.test_pgd_restarts, args.device, early_stop=True)
            delta = delta.detach()
            attack_output = model(clamp(inputs + delta, lower_limit, upper_limit))
        elif args.attack_during_test == 'deepfool':
            pert_inputs, loop, perturbation = deepfool(model, inputs, targets, num_classes=args.deepfool_classes_num, max_iter=args.deepfool_max_iter, device=args.device)
            attack_output = model(pert_inputs)
            df_loop.append(loop.cpu().numpy())
            df_perturbation_norm.append(perturbation.cpu().numpy())

        attack_loss = criterion(attack_output, targets)

        output = model(inputs)
        loss = criterion(output, targets)

        inputs_grad = get_input_grad(model, inputs, targets, delta_init='none', backprop=False, device=args.device)
        # inputs_grad = get_input_grad_v2(model, inputs, targets)
        norm = inputs_grad.view(inputs_grad.shape[0], -1).norm(dim=1)

        test_attack_loss += attack_loss.item() * targets.size(0)
        test_attack_correct += (attack_output.max(1)[1] == targets).sum().item()
        test_standard_loss += loss.item() * targets.size(0)
        test_standard_correct += (output.max(1)[1] == targets).sum().item()
        test_norm.append(norm.cpu().numpy())
        test_total += targets.size(0)
    all_norm = np.concatenate(test_norm)
    if args.attack_during_test == 'deepfool':
        df_loop = np.concatenate(df_loop)
        df_perturbation_norm = np.concatenate(df_perturbation_norm)
    else:
        df_loop = np.array([])
        df_perturbation_norm = np.array([])
    return test_standard_loss / test_total, 100. * test_standard_correct / test_total, test_attack_loss / test_total, 100. * test_attack_correct / test_total, all_norm, df_loop, df_perturbation_norm


def main():
    args = get_args()

    OUTPUT_DIR = './output'
    LOG_DIR = './output/log'
    TENSORBOARD_DIR = './output/tensorboard'
    CHECKPOINT_DIR = './output/checkpoint'
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
    # Model
    args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda'
        cudnn.benchmark = True
    logger.info(args)
    logger.info(f"model trained on {args.device}")

    trainloader, testloader = get_loader(args, logger)

    logger.info('==> Building model..')
    if args.model == 'ResNet18':
        model = ResNet18()
        logger.info("The model used is ResNet18")
    elif args.model == 'PreActResNet18':
        model = PreActResNet18()
        logger.info("The model used is PreActResNet18")
    else:
        logger.info("This model hasn't been defined ", args.model)
        raise NotImplementedError
    if torch.cuda.device_count() > 1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_schedule == 'multistep':
        step_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.lr_change_epoch,
                                                           gamma=0.1)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    args.train_epsilon = (args.train_epsilon / 255.) / std
    args.train_fgsm_alpha = (args.train_fgsm_alpha / 255.) / std
    args.train_pgd_alpha = (args.train_pgd_alpha / 255.) / std
    args.test_epsilon = (args.test_epsilon / 255.) / std
    args.test_pgd_alpha = (args.test_pgd_alpha / 255.) / std
    if args.finetune:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(os.path.join(CHECKPOINT_DIR,
                                           args.resumed_model_name)), f'Error: no asked checkpoint file {args.resumed_model_name} found! '
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name))
        model.load_state_dict(checkpoint['model'])
        log_resumed_info(checkpoint, logger, writer)

    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Test Standard Loss \t Test Standard Acc \t Test Attack Loss \t Test Attack Acc \t Train Norm \t Train Norm Std \t Train Norm Median \t Test Norm \t Test Norm Std \t Test Norm Median')
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        train_loss, train_acc, train_all_norm, train_df_loop, train_df_perturbation_norm = train(args, model, trainloader, optimizer, criterion, epoch)
        train_norm_mean = train_all_norm.mean()
        train_norm_median = np.median(train_all_norm)
        train_norm_std = train_all_norm.std()
        train_time = time.time()

        test_standard_loss, test_standard_acc, test_attack_loss, test_attack_acc, test_all_norm, test_df_loop, test_df_perturbation_norm = test(args, model, testloader, criterion)
        test_norm_mean = test_all_norm.mean()
        test_norm_median = np.median(test_all_norm)
        test_norm_std = test_all_norm.std()
        test_time = time.time()

        logger.info(
            '%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.2f \t \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f \t \t \t %.4f \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
            epoch, train_time - start_time, test_time - train_time, optimizer.param_groups[0]['lr'], train_loss,
            train_acc, test_standard_loss, test_standard_acc, test_attack_loss, test_attack_acc, train_norm_mean,
            train_norm_std, train_norm_median, test_norm_mean, test_norm_std, test_norm_median)

        if test_attack_acc > best_acc:
            save_checkpoint(model, epoch, train_loss, train_acc, test_standard_loss, test_standard_acc,
                            test_attack_loss, test_attack_acc, train_norm_mean, test_norm_mean,
                            os.path.join(CHECKPOINT_DIR, args.exp_name + '_best.pth'))
            best_acc = test_attack_acc

        tb_writer(writer, model, epoch, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_standard_loss,
                  test_standard_acc, test_attack_loss, test_attack_acc, train_norm_mean, train_norm_std, train_norm_median,
                  test_norm_mean, test_norm_std, test_norm_median, train_all_norm, test_all_norm, train_df_loop, train_df_perturbation_norm, test_df_loop, test_df_perturbation_norm)

        if args.lr_schedule == 'multistep':
            step_lr_scheduler.step()
    save_checkpoint(model, epoch, train_loss, train_acc, test_standard_loss, test_standard_acc, test_attack_loss,
                    test_attack_acc, train_norm_mean, test_norm_mean, os.path.join(CHECKPOINT_DIR, args.exp_name + '_final.pth'))
    writer.close()


if __name__ == "__main__":
    main()
