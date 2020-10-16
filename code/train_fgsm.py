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


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset_path', default='./data', help='path of the dataset')

    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', default='multistep', choices=['multistep', 'constant'])
    parser.add_argument('--lr_change_epoch', nargs='+', default=[100, 150], type=int)
    parser.add_argument('--batch_size', '-b', default=128, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)

    parser.add_argument('--train_epsilon', default=8, type=int)
    parser.add_argument('--train_fgsm_alpha', default=10, type=float)
    parser.add_argument('--train_random_start', action='store_true')
    parser.add_argument('--eval_epsilon', default=8, type=int)
    parser.add_argument('--eval_fgsm_alpha', default=10, type=float)
    parser.add_argument('--eval_random_start', action='store_true')
    parser.add_argument('--eval_pgd_alpha', default=2, type=float)
    parser.add_argument('--eval_pgd_attack_iters', default=10, type=int)
    parser.add_argument('--eval_pgd_restarts', default=1, type=int)
    parser.add_argument('--eval_deepfool_classes_num', default=2, type=int)
    parser.add_argument('--eval_deepfool_max_iter', default=50, type=int)

    parser.add_argument('--finetune', action='store_true', help='finetune the pre-trained model with adversarial '
                                                                'samples or regularization')
    parser.add_argument('--resumed_model_name', default='standard_cifar.pth', help='the file name of resumed model')
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()


# Training
def train(args, model, trainloader, optimizer, criterion):
    model.train()
    train_clean_loss, train_clean_correct, train_fgsm_loss, train_fgsm_correct, train_total = 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        delta = attack_pgd(model, inputs, targets, args.train_epsilon, args.train_fgsm_alpha, 1, 1, args.device,
                           random_start=args.train_random_start).detach()

        fgsm_outputs = model(clamp(inputs + delta, lower_limit, upper_limit))
        fgsm_loss = criterion(fgsm_outputs, targets)
        optimizer.zero_grad()
        fgsm_loss.backward()
        optimizer.step()

        clean_outputs = model(inputs)
        clean_loss = criterion(clean_outputs, targets)

        train_fgsm_loss += fgsm_loss.item() * targets.size(0)
        train_fgsm_correct += (fgsm_outputs.max(dim=1)[1] == targets).sum().item()
        train_clean_loss += clean_loss.item() * targets.size(0)
        train_clean_correct += (clean_outputs.max(dim=1)[1] == targets).sum().item()
        train_total += targets.size(0)

    return train_clean_loss / train_total, 100. * train_clean_correct / train_total, train_fgsm_loss / train_total, 100. * train_fgsm_correct / train_total


def eval(args, model, testloader, criterion):
    model.eval()
    test_clean_loss, test_clean_correct, test_fgsm_loss, test_fgsm_correct, test_pgd_loss, test_pgd_correct, test_total = 0, 0, 0, 0, 0, 0, 0
    test_df_loop, test_df_perturbation_norm = 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # clean
        clean_outputs = model(inputs)
        clean_loss = criterion(clean_outputs, targets)

        # fgsm
        fgsm_delta = attack_pgd(model, inputs, targets, args.eval_epsilon, args.eval_fgsm_alpha, 1, 1, args.device,
                           random_start=args.eval_random_start).detach()
        fgsm_outputs = model(clamp(inputs + fgsm_delta, lower_limit, upper_limit))
        fgsm_loss = criterion(fgsm_outputs, targets)

        # pgd
        pgd_delta = attack_pgd(model, inputs, targets, args.eval_epsilon, args.eval_pgd_alpha,
                               args.eval_pgd_attack_iters, args.eval_pgd_restarts, args.device,
                               early_stop=True).detach()
        pgd_outputs = model(clamp(inputs + pgd_delta, lower_limit, upper_limit))
        pgd_loss = criterion(pgd_outputs, targets)

        # deepfool
        pert_inputs, loop, perturbation = deepfool(model, inputs, num_classes=args.eval_deepfool_classes_num,
                                                   max_iter=args.eval_deepfool_max_iter, device=args.device)
        test_clean_loss += clean_loss.item() * targets.size(0)
        test_clean_correct += (clean_outputs.max(dim=1)[1] == targets).sum().item()
        test_fgsm_loss += fgsm_loss.item() * targets.size(0)
        test_fgsm_correct += (fgsm_outputs.max(dim=1)[1] == targets).sum().item()
        test_pgd_loss += pgd_loss.item() * targets.size(0)
        test_pgd_correct += (pgd_outputs.max(dim=1)[1] == targets).sum().item()
        test_total += targets.size(0)
        test_df_loop += loop.sum().item()
        test_df_perturbation_norm += perturbation.sum().item()

    return test_clean_loss / test_total, 100. * test_clean_correct / test_total, \
           test_fgsm_loss / test_total, 100. * test_fgsm_correct / test_total, \
           test_pgd_loss / test_total, 100. * test_pgd_correct / test_total, \
           test_df_loop / test_total, test_df_perturbation_norm / test_total


def tb_writer(writer, epoch, lr, train_clean_loss, train_clean_acc, train_fgsm_loss, train_fgsm_acc,
              test_clean_loss, test_clean_acc, test_fgsm_loss, test_fgsm_acc, test_pgd_loss, test_pgd_acc, test_df_loop, test_df_perturbation_norm):
    writer.add_scalars('loss',
                       {'train_clean': train_clean_loss, 'train_fgsm': train_fgsm_loss,
                        'test_clean': test_clean_loss, 'test_fgsm': test_fgsm_loss, 'test_pgd': test_pgd_loss}, epoch)
    writer.add_scalars('accuracy',
                       {'train_clean': train_clean_acc, 'train_fgsm': train_fgsm_acc,
                        'test_clean': test_clean_acc, 'test_fgsm': test_fgsm_acc, 'test_pgd': test_pgd_acc}, epoch)
    writer.add_scalar('learning rate', lr, epoch)
    writer.add_scalar('test_df_loop', test_df_loop, epoch)
    writer.add_scalar('test_df_perturbation_norm', test_df_perturbation_norm, epoch)


def eval_init(args, writer, logger, model, trainloader, testloader, criterion, opt):
    model.eval()
    train_clean_loss, train_clean_correct, train_fgsm_loss, train_fgsm_correct, train_total = 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        # fgsm
        delta = attack_pgd(model, inputs, targets, args.train_epsilon, args.train_fgsm_alpha, 1, 1, args.device,
                           random_start=args.train_random_start).detach()
        fgsm_outputs = model(clamp(inputs + delta, lower_limit, upper_limit))
        fgsm_loss = criterion(fgsm_outputs, targets)

        clean_outputs = model(inputs)
        clean_loss = criterion(clean_outputs, targets)

        train_fgsm_loss += fgsm_loss.item() * targets.size(0)
        train_fgsm_correct += (fgsm_outputs.max(dim=1)[1] == targets).sum().item()
        train_clean_loss += clean_loss.item() * targets.size(0)
        train_clean_correct += (clean_outputs.max(dim=1)[1] == targets).sum().item()
        train_total += targets.size(0)

    test_clean_loss, test_clean_acc, test_fgsm_loss, test_fgsm_acc, test_pgd_loss, test_pgd_acc, test_df_loop, test_df_perturbation_norm = eval(args, model, testloader, criterion)
    tb_writer(writer, 0, opt.param_groups[0]['lr'],
              train_clean_loss / train_total, 100. * train_clean_correct / train_total, train_fgsm_loss / train_total, 100. * train_fgsm_correct / train_total,
              test_clean_loss, test_clean_acc, test_fgsm_loss, test_fgsm_acc, test_pgd_loss, test_pgd_acc, test_df_loop, test_df_perturbation_norm)
    logger.info(
        '%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.2f \t \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f',
        0, -1, -1, opt.param_groups[0]['lr'], train_fgsm_loss / train_total, 100. * train_fgsm_correct / train_total, test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc)


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

    args.train_epsilon = (args.train_epsilon / 255.) / std
    args.train_fgsm_alpha = (args.train_fgsm_alpha / 255.) / std
    args.eval_epsilon = (args.eval_epsilon / 255.) / std
    args.eval_fgsm_alpha = (args.eval_fgsm_alpha / 255.) / std
    args.eval_pgd_alpha = (args.eval_pgd_alpha / 255.) / std
    if args.finetune:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(os.path.join(CHECKPOINT_DIR,
                                           args.resumed_model_name)), f'Error: no asked checkpoint file {args.resumed_model_name} found! '
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name))
        model.load_state_dict(checkpoint['model'])
        log_resumed_info(checkpoint, logger, writer)

    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Test Standard Loss \t Test Standard '
        'Acc \t Test Attack Loss \t Test Attack Acc')
    eval_init(args, writer, logger, model, trainloader, testloader, criterion, optimizer)

    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_clean_loss, train_clean_acc, train_fgsm_loss, train_fgsm_acc = train(args, model, trainloader, optimizer, criterion)
        train_time = time.time()

        test_clean_loss, test_clean_acc, test_fgsm_loss, test_fgsm_acc, test_pgd_loss, test_pgd_acc, test_df_loop, test_df_perturbation_norm = eval(args, model, testloader, criterion)
        test_time = time.time()

        tb_writer(writer, epoch+1, optimizer.param_groups[0]['lr'],
                  train_clean_loss, train_clean_acc, train_fgsm_loss, train_fgsm_acc,
                  test_clean_loss, test_clean_acc, test_fgsm_loss, test_fgsm_acc, test_pgd_loss, test_pgd_acc, test_df_loop, test_df_perturbation_norm)
        logger.info(
            '%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.2f \t \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f',
            epoch+1, train_time-start_time, test_time-train_time, optimizer.param_groups[0]['lr'],
            train_fgsm_loss, train_fgsm_acc, test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc)
        if args.lr_schedule == 'multistep':
            step_lr_scheduler.step()

    writer.close()


if __name__ == "__main__":
    main()
