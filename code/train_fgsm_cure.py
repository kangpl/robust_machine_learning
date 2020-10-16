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
from cure import cure


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset_path', default='./data', help='path of the dataset')

    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', default='multistep', choices=['multistep', 'constant'])
    parser.add_argument('--lr_change_epoch', nargs='+', default=[100, 150], type=int)
    parser.add_argument('--batch_size', '-b', default=128, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)

    parser.add_argument('--cure_lambda', default=4, type=float)
    parser.add_argument('--cure_h_max', default=3.0, type=float)
    parser.add_argument('--train_epsilon', default=8, type=int)
    parser.add_argument('--train_fgsm_alpha', default=10, type=float)
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
def train(args, model, trainloader, optimizer, criterion, h):
    model.train()
    train_loss = 0
    train_correct = 0
    train_reg = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        delta = attack_pgd(model, inputs, targets, args.train_epsilon, args.train_fgsm_alpha, 1, 1, args.device).detach()
        reg = cure(model, inputs, targets, h=h, device=args.device)

        outputs = model(clamp(inputs + delta, lower_limit, upper_limit))
        loss = criterion(outputs, targets) + args.cure_lambda * reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * targets.size(0)
        train_correct += (outputs.max(dim=1)[1] == targets).sum().item()
        train_total += targets.size(0)
        train_reg += reg.item() * targets.size(0)

    return train_loss / train_total, 100. * train_correct / train_total, train_reg / train_total


def test(args, model, testloader, criterion):
    model.eval()
    test_clean_loss = 0
    test_clean_correct = 0
    test_pgd_loss = 0
    test_pgd_correct = 0
    test_total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # clean
        outputs_clean = model(inputs)
        loss_clean = criterion(outputs_clean, targets)

        # pgd
        delta_pgd = attack_pgd(model, inputs, targets, args.test_epsilon, args.test_pgd_alpha,
                               args.test_pgd_attack_iters, args.test_pgd_restarts, args.device,
                               early_stop=True).detach()
        outputs_pgd = model(clamp(inputs + delta_pgd, lower_limit, upper_limit))
        loss_pgd = criterion(outputs_pgd, targets)

        test_clean_loss += loss_clean.item() * targets.size(0)
        test_clean_correct += (outputs_clean.max(1)[1] == targets).sum().item()
        test_pgd_loss += loss_pgd.item() * targets.size(0)
        test_pgd_correct += (outputs_pgd.max(1)[1] == targets).sum().item()
        test_total += targets.size(0)

    return test_clean_loss / test_total, 100. * test_clean_correct / test_total, test_pgd_loss / test_total, 100. * test_pgd_correct / test_total


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
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_schedule == 'multistep':
        step_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.lr_change_epoch,
                                                           gamma=0.1)

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    args.train_epsilon = (args.train_epsilon / 255.) / std
    args.train_fgsm_alpha = (args.train_fgsm_alpha / 255.) / std
    args.test_epsilon = (args.test_epsilon / 255.) / std
    args.test_pgd_alpha = (args.test_pgd_alpha / 255.) / std
    if args.finetune:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(os.path.join(CHECKPOINT_DIR,
                                           args.resumed_model_name)), f'Error: no asked checkpoint file {args.resumed_model_name} found! '
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name))
        model.load_state_dict(checkpoint['model'])
        log_resumed_info(checkpoint, logger)

    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train CURE \t Test Standard Loss \t Test Standard Acc \t Test Attack Loss \t Test Attack Acc')

    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        train_loss, train_acc, train_reg = train(args, model, trainloader, optimizer, criterion, args.cure_h_max)
        train_time = time.time()

        test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc = test(args, model, testloader, criterion)
        test_time = time.time()

        logger.info(
            '%d \t %.1f \t \t %.1f \t \t %.6f \t %.4f \t %.2f \t \t %.4f \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f',
            epoch, train_time - start_time, test_time - train_time, optimizer.param_groups[0]['lr'], train_loss,
            train_acc, train_reg, test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc)

        tb_writer_cure(writer, epoch, optimizer.param_groups[0]['lr'], train_loss, train_acc, train_reg, test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc)

        if args.lr_schedule == 'multistep':
            step_lr_scheduler.step()
    writer.close()


if __name__ == "__main__":
    main()