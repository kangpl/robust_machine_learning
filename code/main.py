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

    parser.add_argument('--attack_during_train', default='fgsm', type=str, choices=['pgd', 'fgsm', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--fgsm_alpha', default=10, type=float)
    parser.add_argument('--pgd_alpha', default=2, type=float)
    parser.add_argument('--attack_iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)

    parser.add_argument('--finetune', action='store_true', help='finetune the pre-trained model with adversarial '
                                                                'samples or regularization')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
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
            delta = attack_pgd(model, inputs, targets, args.epsilon, args.pgd_alpha, args.attack_iters, args.restarts,
                               args.device)
        elif args.attack_during_train == 'fgsm':
            delta = attack_pgd(model, inputs, targets, args.epsilon, args.fgsm_alpha, 1, 1, args.device)
        elif args.attack_during_train == 'none':
            delta = torch.zeros_like(inputs)
        delta = delta.detach()

        outputs = model(clamp(inputs + delta, lower_limit, upper_limit))
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * targets.size(0)
        train_correct += (outputs.max(dim=1)[1] == targets).sum().item()
        train_total += targets.size(0)
    return train_loss / train_total, 100. * train_correct / train_total


def test(args, model, testloader, criterion):
    model.eval()
    test_standard_loss = 0
    test_standard_correct = 0
    test_attack_loss = 0
    test_attack_correct = 0
    test_total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        delta = attack_pgd(model, inputs, targets, args.epsilon, args.pgd_alpha, args.attack_iters, args.restarts,
                           args.device)
        delta = delta.detach()

        attack_output = model(clamp(inputs + delta, lower_limit, upper_limit))
        attack_loss = criterion(attack_output, targets)

        output = model(inputs)
        loss = criterion(output, targets)

        test_attack_loss += attack_loss.item() * targets.size(0)
        test_attack_correct += (attack_output.max(1)[1] == targets).sum().item()
        test_standard_loss += loss.item() * targets.size(0)
        test_standard_correct += (output.max(1)[1] == targets).sum().item()
        test_total += targets.size(0)

    return test_standard_loss / test_total, 100. * test_standard_correct / test_total, test_attack_loss / test_total, 100. * test_attack_correct / test_total


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
    args.epsilon = (args.epsilon / 255.) / std
    args.fgsm_alpha = (args.fgsm_alpha / 255.) / std
    args.pgd_alpha = (args.pgd_alpha / 255.) / std
    if args.resume or args.finetune:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(os.path.join(CHECKPOINT_DIR,
                                           args.resumed_model_name)), f'Error: no asked checkpoint file {args.resumed_model_name} found! '
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name))
        model.load_state_dict(checkpoint['model'])
        resumed_epoch = checkpoint['epoch']
        resumed_train_loss = checkpoint['train_loss']
        resumed_train_acc = checkpoint['train_acc']
        resumed_test_standard_loss = checkpoint['test_standard_loss']
        resumed_test_standard_acc = checkpoint['test_standard_acc']
        resumed_test_attack_loss = checkpoint['test_attack_loss']
        resumed_test_attack_acc = checkpoint['test_attack_acc']
        logger.info(
            f"finetune from epoch {resumed_epoch}, train loss {resumed_train_loss}, train acc {resumed_train_acc}, "
            f"Test Standard Loss {resumed_test_standard_loss}, Test Standard Acc {resumed_test_standard_acc}, "
            f"Test Attack Loss {resumed_test_attack_loss}, Test Attack Acc {resumed_test_attack_acc}")
        if args.resume:
            best_acc = checkpoint['test_attack_acc']
            start_epoch = checkpoint['epoch']
        writer.add_scalars('loss', {'train': resumed_train_loss, 'test_standard': resumed_test_standard_loss,
                                    "test_attack": resumed_test_attack_loss}, start_epoch)
        writer.add_scalars('accuracy', {'train': resumed_train_acc, 'test_standard': resumed_test_standard_acc,
                                        'test_attack': resumed_test_attack_acc}, start_epoch)

    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Test Standard Loss \t Test Standard '
        'Acc \t Test Robust Loss \t Test Robust Acc')
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(args, model, trainloader, optimizer, criterion)
        train_time = time.time()
        test_standard_loss, test_standard_acc, test_attack_loss, test_attack_acc = test(args, model, testloader,
                                                                                        criterion)
        test_time = time.time()

        logger.info(
            '%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.2f \t \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f',
            epoch, train_time - start_time, test_time - train_time, optimizer.param_groups[0]['lr'], train_loss,
            train_acc, test_standard_loss, test_standard_acc, test_attack_loss, test_attack_acc)

        if test_attack_acc > best_acc:
            state = {
                'model': model.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_standard_loss': test_standard_loss,
                'test_standard_acc': test_standard_acc,
                'test_attack_loss': test_attack_loss,
                'test_attack_acc': test_attack_acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(CHECKPOINT_DIR, args.exp_name + '_best.pth'))
            best_acc = test_attack_acc

        writer.add_scalars('loss',
                           {'train': train_loss, 'test_standard': test_standard_loss, 'test_attack': test_attack_loss},
                           epoch + 1)
        writer.add_scalars('accuracy',
                           {'train': train_acc, 'test_standard': test_standard_acc, 'test_attack': test_attack_acc},
                           epoch + 1)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch + 1)
        if args.lr_schedule == 'multistep':
            step_lr_scheduler.step()

    torch.save({
        'model': model.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_standard_loss': test_standard_loss,
        'test_standard_acc': test_standard_acc,
        'test_attack_loss': test_attack_loss,
        'test_attack_acc': test_attack_acc,
        'epoch': epoch},
        os.path.join(CHECKPOINT_DIR, args.exp_name + '_final.pth'))
    writer.close()


if __name__ == "__main__":
    main()
