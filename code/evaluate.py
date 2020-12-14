"""Train CIFAR10 with PyTorch."""

import argparse
import logging
import os

import torch.nn as nn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18
from utils.util import *


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset_path', default='./data', help='path of the dataset')

    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--eval_pgd_ratio', default=0.25, type=float)
    parser.add_argument('--eval_pgd_attack_iters', default=50, type=int)
    parser.add_argument('--eval_pgd_restarts', default=10, type=int)
    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--batch_size', '-b', default=256, type=int)
    parser.add_argument('--resumed_model_name', default='standard_cifar.pth', help='the file name of resumed model')
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()


def eval(args, model, testloader, criterion):
    model.eval()
    test_clean_loss, test_clean_correct, test_pgd_loss, test_pgd_correct, test_pgd_delta_norm, test_total = 0, 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # clean
        clean_outputs = model(inputs)
        clean_loss = criterion(clean_outputs, targets)

        # pgd
        pgd_delta = attack_pgd(model, inputs, targets, args.epsilon, args.eval_pgd_ratio * args.epsilon, args.eval_pgd_attack_iters, args.eval_pgd_restarts, args.device, early_stop=True).detach()
        pgd_outputs = model(clamp(inputs + pgd_delta, lower_limit, upper_limit))
        pgd_loss = criterion(pgd_outputs, targets)
        pgd_delta_norm = pgd_delta.view(pgd_delta.shape[0], -1).norm(dim=1)

        test_clean_loss += clean_loss.item() * targets.size(0)
        test_clean_correct += (clean_outputs.max(dim=1)[1] == targets).sum().item()
        test_pgd_loss += pgd_loss.item() * targets.size(0)
        test_pgd_correct += (pgd_outputs.max(dim=1)[1] == targets).sum().item()
        test_pgd_delta_norm += pgd_delta_norm.sum().item()
        test_total += targets.size(0)

    return test_clean_loss / test_total, 100. * test_clean_correct / test_total, \
           test_pgd_loss / test_total, 100. * test_pgd_correct / test_total, test_pgd_delta_norm / test_total


def main():
    args = get_args()

    OUTPUT_DIR = './output'
    LOG_DIR = './output/log'
    CHECKPOINT_DIR = '../../../../scratch/pekang/checkpoint/'
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)


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

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    testset = datasets.CIFAR10(
        root=args.dataset_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

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
    args.epsilon = (args.epsilon / 255.) / std

    # Evaluate best and final
    logger.info('best')
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name + f'_best.pth'))
    model.load_state_dict(checkpoint['model'])
    log_resumed_info(checkpoint, logger)
    best_clean_loss, best_clean_acc, best_pgd_loss, best_pgd_acc, best_pgd_delta_norm = eval(args, model, testloader, criterion)
    logger.info('%d \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f \t %.2f', checkpoint['epoch'], best_clean_loss, best_clean_acc, best_pgd_loss, best_pgd_acc, best_pgd_delta_norm)

    logger.info('final')
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name + f'_final.pth'))
    model.load_state_dict(checkpoint['model'])
    log_resumed_info(checkpoint, logger)
    final_clean_loss, final_clean_acc, final_pgd_loss, final_pgd_acc, final_pgd_delta_norm = eval(args, model, testloader, criterion)
    logger.info('%d \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f \t %.2f', checkpoint['epoch'], final_clean_loss, final_clean_acc, final_pgd_loss, final_pgd_acc, final_pgd_delta_norm)


if __name__ == "__main__":
    main()
