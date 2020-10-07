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
    parser.add_argument('--attack_during_train', default='fgsm', type=str, choices=['pgd', 'fgsm', 'none', 'both'])
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
def train(args, model, trainloader, optimizer, criterion):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

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

        train_loss += loss.item() * targets.size(0)
        train_correct += (outputs.max(dim=1)[1] == targets).sum().item()
        train_total += targets.size(0)

    return train_loss / train_total, 100. * train_correct / train_total


def test(args, model, trainloader, testloader, criterion):
    model.eval()
    train_pgd_loss = 0
    train_pgd_correct = 0
    train_deepfool_loss = 0
    train_deepfool_correct = 0
    train_df_loop = []
    train_df_perturbation_norm = []
    train_df_grad_norm = []
    train_input_grad_norm = []
    train_cos = []
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx % 10 == 0:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # pgd
            delta_pgd = attack_pgd(model, inputs, targets, args.test_epsilon, args.test_pgd_alpha,
                                   args.test_pgd_attack_iters, args.test_pgd_restarts, args.device,
                                   early_stop=True).detach()
            outputs_pgd = model(clamp(inputs + delta_pgd, lower_limit, upper_limit))
            loss_pgd = criterion(outputs_pgd, targets)

            # deepfool attack
            pert_inputs, loop, perturbation = deepfool(model, inputs, targets, num_classes=args.deepfool_classes_num,
                                                       max_iter=args.deepfool_max_iter, device=args.device)
            deepfool_grad, outputs_deepfool, loss_deepfool = get_input_grad_v2(model, pert_inputs, targets)
            deepfool_grad_norm = deepfool_grad.view(deepfool_grad.shape[0], -1).norm(dim=1)
            loop_nz_idx = (loop != 0)
            loop_valid = loop[loop_nz_idx]
            perturbation_valid = perturbation[loop_nz_idx]
            deepfool_grad_norm_valid = deepfool_grad_norm[loop_nz_idx]

            # input gradient norm
            inputs_grad, _, _ = get_input_grad_v2(model, inputs, targets)
            inputs_grad_norm = inputs_grad.view(inputs_grad.shape[0], -1).norm(dim=1)
            inputs_grad_norm_valid = inputs_grad_norm[loop_nz_idx]

            # calculate cosine
            grads_nnz_idx = (inputs_grad_norm != 0) * (deepfool_grad_norm != 0) * loop_nz_idx
            grad1, grad2 = inputs_grad[grads_nnz_idx], deepfool_grad[grads_nnz_idx]
            grad1_norms, grad2_norms = inputs_grad_norm[grads_nnz_idx], deepfool_grad_norm[grads_nnz_idx]
            grad1_normalized = grad1 / grad1_norms[:, None, None, None]
            grad2_normalized = grad2 / grad2_norms[:, None, None, None]
            cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))

            train_pgd_loss += loss_pgd.item() * targets.size(0)
            train_pgd_correct += (outputs_pgd.max(dim=1)[1] == targets).sum().item()
            train_deepfool_loss += loss_deepfool.item() * targets.size(0)
            train_deepfool_correct += (outputs_deepfool.max(dim=1)[1] == targets).sum().item()
            train_total += targets.size(0)
            train_df_loop.append(loop_valid.cpu().numpy())
            train_df_perturbation_norm.append(perturbation_valid.cpu().numpy())
            train_df_grad_norm.append(deepfool_grad_norm_valid.cpu().numpy())
            train_input_grad_norm.append(inputs_grad_norm_valid.cpu().numpy())
            train_cos.append(cos.cpu().numpy())
    train_df_loop = np.concatenate(train_df_loop)
    train_df_perturbation_norm = np.concatenate(train_df_perturbation_norm)
    train_df_grad_norm = np.concatenate(train_df_grad_norm)
    train_input_grad_norm = np.concatenate(train_input_grad_norm)
    train_cos = np.concatenate(train_cos)

    test_clean_loss = 0
    test_clean_correct = 0
    test_fgsm_loss = 0
    test_fgsm_correct = 0
    test_pgd_loss = 0
    test_pgd_correct = 0
    test_deepfool_loss = 0
    test_deepfool_correct = 0
    test_df_loop = []
    test_df_perturbation_norm = []
    test_df_grad_norm = []
    test_input_grad_norm = []
    test_cos = []
    test_total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # clean
        outputs_clean = model(inputs)
        loss_clean = criterion(outputs_clean, targets)

        # fgsm
        delta_fgsm = attack_pgd(model, inputs, targets, args.test_epsilon, args.train_fgsm_alpha, 1, 1,
                                args.device).detach()
        outputs_fgsm = model(clamp(inputs + delta_fgsm, lower_limit, upper_limit))
        loss_fgsm = criterion(outputs_fgsm, targets)

        # pgd
        delta_pgd = attack_pgd(model, inputs, targets, args.test_epsilon, args.test_pgd_alpha,
                               args.test_pgd_attack_iters, args.test_pgd_restarts, args.device,
                               early_stop=True).detach()
        outputs_pgd = model(clamp(inputs + delta_pgd, lower_limit, upper_limit))
        loss_pgd = criterion(outputs_pgd, targets)

        # deepfool
        pert_inputs, loop, perturbation = deepfool(model, inputs, targets, num_classes=args.deepfool_classes_num,
                                                   max_iter=args.deepfool_max_iter, device=args.device)
        deepfool_grad, outputs_deepfool, loss_deepfool = get_input_grad_v2(model, pert_inputs, targets)
        deepfool_grad_norm = deepfool_grad.view(deepfool_grad.shape[0], -1).norm(dim=1)
        loop_nz_idx = (loop != 0)
        loop_valid = loop[loop_nz_idx]
        perturbation_valid = perturbation[loop_nz_idx]
        deepfool_grad_norm_valid = deepfool_grad_norm[loop_nz_idx]

        # norm
        inputs_grad, _, _ = get_input_grad_v2(model, inputs, targets)
        inputs_grad_norm = inputs_grad.view(inputs_grad.shape[0], -1).norm(dim=1)
        inputs_grad_norm_valid = inputs_grad_norm[loop_nz_idx]

        # calculate cosine
        grads_nnz_idx = (inputs_grad_norm != 0) * (deepfool_grad_norm != 0) * loop_nz_idx
        grad1, grad2 = inputs_grad[grads_nnz_idx], deepfool_grad[grads_nnz_idx]
        grad1_norms, grad2_norms = inputs_grad_norm[grads_nnz_idx], deepfool_grad_norm[grads_nnz_idx]
        grad1_normalized = grad1 / grad1_norms[:, None, None, None]
        grad2_normalized = grad2 / grad2_norms[:, None, None, None]
        cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))

        test_clean_loss += loss_clean.item() * targets.size(0)
        test_clean_correct += (outputs_clean.max(1)[1] == targets).sum().item()
        test_fgsm_loss += loss_fgsm.item() * targets.size(0)
        test_fgsm_correct += (outputs_fgsm.max(1)[1] == targets).sum().item()
        test_pgd_loss += loss_pgd.item() * targets.size(0)
        test_pgd_correct += (outputs_pgd.max(1)[1] == targets).sum().item()
        test_deepfool_loss += loss_deepfool.item() * targets.size(0)
        test_deepfool_correct += (outputs_deepfool.max(1)[1] == targets).sum().item()
        test_total += targets.size(0)
        test_df_loop.append(loop_valid.cpu().numpy())
        test_df_perturbation_norm.append(perturbation_valid.cpu().numpy())
        test_df_grad_norm.append(deepfool_grad_norm_valid.cpu().numpy())
        test_input_grad_norm.append(inputs_grad_norm_valid.cpu().numpy())
        test_cos.append(cos.cpu().numpy())
    test_df_loop = np.concatenate(test_df_loop)
    test_df_perturbation_norm = np.concatenate(test_df_perturbation_norm)
    test_df_grad_norm = np.concatenate(test_df_grad_norm)
    test_input_grad_norm = np.concatenate(test_input_grad_norm)
    test_cos = np.concatenate(test_cos)

    return train_pgd_loss / train_total, 100. * train_pgd_correct / train_total, \
           train_deepfool_loss / train_total, 100. * train_deepfool_correct / train_total, \
           train_input_grad_norm, train_df_loop, train_df_perturbation_norm, train_df_grad_norm, train_cos, \
           test_clean_loss / test_total, 100. * test_clean_correct / test_total, \
           test_fgsm_loss / test_total, 100. * test_fgsm_correct / test_total, \
           test_pgd_loss / test_total, 100. * test_pgd_correct / test_total, \
           test_deepfool_loss / test_total, 100. * test_deepfool_correct / test_total, \
           test_input_grad_norm, test_df_loop, test_df_perturbation_norm, test_df_grad_norm, test_cos


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
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Test Standard Loss \t Test Standard Acc \t Test Attack Loss \t Test Attack Acc')
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        train_fgsm_loss, train_fgsm_acc = train(args, model, trainloader, optimizer, criterion)
        train_time = time.time()

        train_pgd_loss, train_pgd_acc, train_deepfool_loss, train_deepfool_acc, \
        train_input_grad_norm, train_df_loop, train_df_perturbation_norm, train_df_grad_norm, train_cos, \
        test_clean_loss, test_clean_acc, test_fgsm_loss, test_fgsm_acc, \
        test_pgd_loss, test_pgd_acc, test_deepfool_loss, test_deepfool_acc, \
        test_input_grad_norm, test_df_loop, test_df_perturbation_norm, test_df_grad_norm, test_cos = test(args, model,
                                                                                                          trainloader,
                                                                                                          testloader,
                                                                                                          criterion)
        test_time = time.time()

        logger.info(
            '%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.2f \t \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f',
            epoch, train_time - start_time, test_time - train_time, optimizer.param_groups[0]['lr'], train_fgsm_loss,
            train_fgsm_acc, test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc)

        if test_pgd_acc > best_acc:
            save_checkpoint(model, epoch, train_fgsm_loss, train_fgsm_acc, test_clean_loss, test_clean_acc,
                            test_pgd_loss, test_pgd_acc, os.path.join(CHECKPOINT_DIR, args.exp_name + '_best.pth'))
            best_acc = test_pgd_acc

        tb_writer(writer, epoch, optimizer.param_groups[0]['lr'], train_fgsm_loss, train_fgsm_acc,
                  train_pgd_loss, train_pgd_acc, train_deepfool_loss, train_deepfool_acc,
                  train_input_grad_norm, train_df_loop, train_df_perturbation_norm, train_df_grad_norm, train_cos,
                  test_clean_loss, test_clean_acc, test_fgsm_loss, test_fgsm_acc,
                  test_pgd_loss, test_pgd_acc, test_deepfool_loss, test_deepfool_acc,
                  test_input_grad_norm, test_df_loop, test_df_perturbation_norm, test_df_grad_norm, test_cos)

        if args.lr_schedule == 'multistep':
            step_lr_scheduler.step()
    save_checkpoint(model, epoch, train_fgsm_loss, train_fgsm_acc, test_clean_loss, test_clean_acc, test_pgd_loss,
                    test_pgd_acc, os.path.join(CHECKPOINT_DIR, args.exp_name + '_final.pth'))
    writer.close()


if __name__ == "__main__":
    main()
