"""Train CIFAR10 with PyTorch."""

import argparse
import logging
import os

import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18
from deepfool import *


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset_path', default='./data', help='path of the dataset')

    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', '-b', default=256, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)

    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--train_fgsm_ratio', default=1, type=float)
    parser.add_argument('--eval_pgd_ratio', default=0.25, type=float)
    parser.add_argument('--eval_pgd_attack_iters', default=10, type=int)
    parser.add_argument('--eval_pgd_restarts', default=1, type=int)

    parser.add_argument('--finetune', action='store_true', help='finetune the pre-trained model with adversarial '
                                                                'samples or regularization')
    parser.add_argument('--resumed_model_name', default='standard_cifar.pth', help='the file name of resumed model')
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()

# Training
def train(args, logger, writer, model, trainloader, testloader, optimizer, criterion):
    global iteration
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        model.train()
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        delta = torch.zeros_like(inputs).to(args.device)
        delta.requires_grad = True
        output = model(inputs + delta)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        grad = delta.grad.detach()
        fgsm_dir = args.train_fgsm_ratio * args.epsilon * torch.sign(grad)

        fgsm_delta = clamp(fgsm_dir, -args.epsilon, args.epsilon)
        fgsm_delta = clamp(fgsm_delta, lower_limit - inputs, upper_limit - inputs).detach()
        fgsm_delta_norm = fgsm_delta.view(fgsm_delta.shape[0], -1).norm(dim=1)

        fgsm_delta_20 = clamp(0.2 * fgsm_dir, -args.epsilon, args.epsilon)
        fgsm_delta_20 = clamp(fgsm_delta_20, lower_limit - inputs, upper_limit - inputs).detach()
        fgsm_20_outputs = model(inputs + fgsm_delta_20)
        fgsm_20_loss = criterion(fgsm_20_outputs, targets)

        fgsm_delta_60 = clamp(0.6 * fgsm_dir, -args.epsilon, args.epsilon)
        fgsm_delta_60 = clamp(fgsm_delta_60, lower_limit - inputs, upper_limit - inputs).detach()
        fgsm_60_outputs = model(inputs + fgsm_delta_60)
        fgsm_60_loss = criterion(fgsm_60_outputs, targets)

        clean_outputs = model(inputs)
        clean_loss = criterion(clean_outputs, targets)

        fgsm_outputs = model(inputs + fgsm_delta)
        fgsm_loss = criterion(fgsm_outputs, targets)
        optimizer.zero_grad()
        fgsm_loss.backward()
        optimizer.step()

        train_fgsm_loss = fgsm_loss.item()
        train_fgsm_acc = 100. * (fgsm_outputs.max(dim=1)[1] == targets).sum().item() / targets.size(0)
        train_clean_loss = clean_loss.item()
        train_clean_acc = 100. * (clean_outputs.max(dim=1)[1] == targets).sum().item() / targets.size(0)
        train_fgsm_delta_norm = fgsm_delta_norm.mean().item()
        train_fgsm_20_loss = fgsm_20_loss.item()
        train_fgsm_20_acc = 100. * (fgsm_20_outputs.max(dim=1)[1] == targets).sum().item() / targets.size(0)
        train_fgsm_60_loss = fgsm_60_loss.item()
        train_fgsm_60_acc = 100. * (fgsm_60_outputs.max(dim=1)[1] == targets).sum().item() / targets.size(0)

        metrics = eval(args, model, testloader, criterion)

        tb_writer(writer, iteration, optimizer.param_groups[0]['lr'],
                  train_clean_loss, train_clean_acc, train_fgsm_loss, train_fgsm_acc, train_fgsm_delta_norm, train_fgsm_20_loss, train_fgsm_20_acc, train_fgsm_60_loss, train_fgsm_60_acc,
                  metrics)
        logger.info(
            '%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.2f \t \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f \t %.2f \t %.2f',
            iteration, -1, -1, optimizer.param_groups[0]['lr'],
            train_fgsm_loss, train_fgsm_acc, metrics['test_clean_loss'], metrics['test_clean_correct'], metrics['test_pgd10_loss'], metrics['test_pgd10_correct'],
            train_fgsm_delta_norm, metrics['test_pgd10_delta_norm'])
        iteration = iteration + 1


def eval(args, model, testloader, criterion):
    model.eval()
    metrics = {'test_clean_loss': 0,
               'test_clean_correct': 0,
               'test_fgsm_loss': 0,
               'test_fgsm_correct': 0,
               'test_fgsm_delta_norm': 0,
               'test_pgd10_loss': 0,
               'test_pgd10_correct': 0,
               'test_pgd10_delta_norm': 0,
               'test_fgsm_20_loss': 0,
               'test_fgsm_20_correct': 0,
               'test_fgsm_60_loss': 0,
               'test_fgsm_60_correct': 0,
               'test1_w_clamp_loss': 0,
               'test1_w_clamp_correct': 0,
               'test1_wo_clamp_loss': 0,
               'test1_wo_clamp_correct': 0,
               'test2_loss': 0,
               'test2_correct': 0,
               'test_rs_w_clamp_loss': 0,
               'test_rs_w_clamp_correct': 0,
               'test_rs_wo_clamp_loss': 0,
               'test_rs_wo_clamp_correct': 0,
               'test_diff_rs_w_clamp_loss': 0,
               'test_diff_rs_w_clamp_correct': 0,
               'test_diff_rs_wo_clamp_loss': 0,
               'test_diff_rs_wo_clamp_correct': 0,
               'test_input_grad_norm': 0,
               'test_df50_loop': 0,
               'test_df50_perturbation_norm': 0,
               'test_total': 0}
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # clean
        input_grad, clean_outputs, clean_loss = get_input_grad_v2(model, inputs, targets)
        input_grad_norm = input_grad.view(input_grad.shape[0], -1).norm(dim=1)
        metrics['test_clean_loss'] += clean_loss.item() * targets.size(0)
        metrics['test_clean_correct'] += (clean_outputs.max(dim=1)[1] == targets).sum().item()
        metrics['test_input_grad_norm'] += input_grad_norm.sum().item()

        # fgsm
        zero = torch.zeros_like(inputs).to(args.device)
        zero.requires_grad = True
        output = model(inputs + zero)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        grad_input = zero.grad.detach()
        fgsm_dir = args.train_fgsm_ratio * args.epsilon * torch.sign(grad_input)

        fgsm_delta = clamp(fgsm_dir, -args.epsilon, args.epsilon)
        fgsm_delta = clamp(fgsm_delta, lower_limit - inputs, upper_limit - inputs).detach()
        fgsm_delta_norm = fgsm_delta.view(fgsm_delta.shape[0], -1).norm(dim=1)
        fgsm_outputs = model(inputs + fgsm_delta)
        fgsm_loss = criterion(fgsm_outputs, targets)
        metrics['test_fgsm_loss'] += fgsm_loss.item() * targets.size(0)
        metrics['test_fgsm_correct'] += (fgsm_outputs.max(dim=1)[1] == targets).sum().item()
        metrics['test_fgsm_delta_norm'] += fgsm_delta_norm.sum().item()

        fgsm_delta_20 = clamp(0.2 * fgsm_dir, -args.epsilon, args.epsilon)
        fgsm_delta_20 = clamp(fgsm_delta_20, lower_limit - inputs, upper_limit - inputs).detach()
        fgsm_20_outputs = model(inputs + fgsm_delta_20)
        fgsm_20_loss = criterion(fgsm_20_outputs, targets)
        metrics['test_fgsm_20_loss'] += fgsm_20_loss.item() * targets.size(0)
        metrics['test_fgsm_20_correct'] += (fgsm_20_outputs.max(dim=1)[1] == targets).sum().item()

        fgsm_delta_60 = clamp(0.6 * fgsm_dir, -args.epsilon, args.epsilon)
        fgsm_delta_60 = clamp(fgsm_delta_60, lower_limit - inputs, upper_limit - inputs).detach()
        fgsm_60_outputs = model(inputs + fgsm_delta_60)
        fgsm_60_loss = criterion(fgsm_60_outputs, targets)
        metrics['test_fgsm_60_loss'] += fgsm_60_loss.item() * targets.size(0)
        metrics['test_fgsm_60_correct'] += (fgsm_60_outputs.max(dim=1)[1] == targets).sum().item()

        delta = zero.detach()
        for i in range(len(args.epsilon)):
            delta[:, i, :, :].uniform_(-args.epsilon[i][0][0].item(), args.epsilon[i][0][0].item())
        delta = clamp(delta, lower_limit - inputs, upper_limit - inputs)

        test1_w_clamp = clamp(delta + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_input), -args.epsilon,
                           args.epsilon)
        test1_w_clamp = clamp(test1_w_clamp, lower_limit - inputs, upper_limit - inputs).detach()
        output_test1_w_clamp = model(inputs + test1_w_clamp)
        loss_test1_w_clamp = F.cross_entropy(output_test1_w_clamp, targets)
        metrics['test1_w_clamp_loss'] += loss_test1_w_clamp.item() * targets.size(0)
        metrics['test1_w_clamp_correct'] += (output_test1_w_clamp.max(dim=1)[1] == targets).sum().item()

        test1_wo_clamp = delta + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_input)
        test1_wo_clamp = clamp(test1_wo_clamp, lower_limit - inputs, upper_limit - inputs).detach()
        output_test1_wo_clamp = model(inputs + test1_wo_clamp)
        loss_test1_wo_clamp = F.cross_entropy(output_test1_wo_clamp, targets)
        metrics['test1_wo_clamp_loss'] += loss_test1_wo_clamp.item() * targets.size(0)
        metrics['test1_wo_clamp_correct'] += (output_test1_wo_clamp.max(dim=1)[1] == targets).sum().item()

        delta.requires_grad = True
        output = model(inputs + delta)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        grad_random = delta.grad.detach()

        fgsm_test2 = clamp(args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random), -args.epsilon, args.epsilon)
        fgsm_test2 = clamp(fgsm_test2, lower_limit - inputs, upper_limit - inputs).detach()
        output_test2 = model(inputs + fgsm_test2)
        loss_test2 = F.cross_entropy(output_test2, targets)
        metrics['test2_loss'] += loss_test2.item() * targets.size(0)
        metrics['test2_correct'] += (output_test2.max(dim=1)[1] == targets).sum().item()

        fgsm_rs_w_clamp = clamp(delta + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random), -args.epsilon,
                        args.epsilon)
        fgsm_rs_w_clamp = clamp(fgsm_rs_w_clamp, lower_limit - inputs, upper_limit - inputs).detach()
        output_rs_w_clamp = model(inputs + fgsm_rs_w_clamp)
        loss_rs_w_clamp = F.cross_entropy(output_rs_w_clamp, targets)
        metrics['test_rs_w_clamp_loss'] += loss_rs_w_clamp.item() * targets.size(0)
        metrics['test_rs_w_clamp_correct'] += (output_rs_w_clamp.max(dim=1)[1] == targets).sum().item()

        fgsm_rs_wo_clamp = delta + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random)
        fgsm_rs_wo_clamp = clamp(fgsm_rs_wo_clamp, lower_limit - inputs, upper_limit - inputs).detach()
        output_rs_wo_clamp = model(inputs + fgsm_rs_wo_clamp)
        loss_rs_wo_clamp = F.cross_entropy(output_rs_wo_clamp, targets)
        metrics['test_rs_wo_clamp_loss'] += loss_rs_wo_clamp.item() * targets.size(0)
        metrics['test_rs_wo_clamp_correct'] += (output_rs_wo_clamp.max(dim=1)[1] == targets).sum().item()

        delta2 = torch.zeros_like(inputs).to(args.device)
        for i in range(len(args.epsilon)):
            delta2[:, i, :, :].uniform_(-args.epsilon[i][0][0].item(), args.epsilon[i][0][0].item())
        delta2 = clamp(delta2, lower_limit - inputs, upper_limit - inputs)
        fgsm_rs_diff_w_clamp = clamp(delta2 + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random), -args.epsilon,
                             args.epsilon)
        fgsm_rs_diff_w_clamp = clamp(fgsm_rs_diff_w_clamp, lower_limit - inputs, upper_limit - inputs).detach()
        output_rs_diff_w_clamp = model(inputs + fgsm_rs_diff_w_clamp)
        loss_rs_diff_w_clamp = F.cross_entropy(output_rs_diff_w_clamp, targets)
        metrics['test_diff_rs_w_clamp_loss'] += loss_rs_diff_w_clamp.item() * targets.size(0)
        metrics['test_diff_rs_w_clamp_correct'] += (output_rs_diff_w_clamp.max(dim=1)[1] == targets).sum().item()

        fgsm_rs_diff_wo_clamp = delta2 + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random)
        fgsm_rs_diff_wo_clamp = clamp(fgsm_rs_diff_wo_clamp, lower_limit - inputs, upper_limit - inputs).detach()
        output_rs_diff_wo_clamp = model(inputs + fgsm_rs_diff_wo_clamp)
        loss_rs_diff_wo_clamp = F.cross_entropy(output_rs_diff_wo_clamp, targets)
        metrics['test_diff_rs_wo_clamp_loss'] += loss_rs_diff_wo_clamp.item() * targets.size(0)
        metrics['test_diff_rs_wo_clamp_correct'] += (output_rs_diff_wo_clamp.max(dim=1)[1] == targets).sum().item()

        metrics['test_total'] += targets.size(0)

        # pgd
        pgd_delta = attack_pgd(model, inputs, targets, args.epsilon, args.eval_pgd_ratio * args.epsilon,
                                   args.eval_pgd_attack_iters, args.eval_pgd_restarts, args.device,
                                   early_stop=True).detach()
        pgd_outputs = model(clamp(inputs + pgd_delta, lower_limit, upper_limit))
        pgd_loss = criterion(pgd_outputs, targets)
        pgd_delta_norm = pgd_delta.view(pgd_delta.shape[0], -1).norm(dim=1)
        metrics['test_pgd10_loss'] += pgd_loss.item() * targets.size(0)
        metrics['test_pgd10_correct'] += (pgd_outputs.max(dim=1)[1] == targets).sum().item()
        metrics['test_pgd10_delta_norm'] += pgd_delta_norm.sum().item()

        # deepfool
        loop, perturbation = deepfool_train(model, inputs, overshoot=0.02, max_iter=50, norm_dist='l_2',
                                            device=args.device, random_start=False, early_stop=True)
        perturbation_norm = perturbation.view(perturbation.shape[0], -1).norm(dim=1)
        metrics['test_df50_loop'] += loop.sum().item()
        metrics['test_df50_perturbation_norm'] += perturbation_norm.sum().item()

    metrics['test_clean_loss'] = metrics['test_clean_loss'] / metrics['test_total']
    metrics['test_clean_correct'] = metrics['test_clean_correct'] / metrics['test_total']
    metrics['test_input_grad_norm'] = metrics['test_input_grad_norm'] / metrics['test_total']
    metrics['test_fgsm_loss'] = metrics['test_fgsm_loss'] / metrics['test_total']
    metrics['test_fgsm_correct'] = metrics['test_fgsm_correct'] / metrics['test_total']
    metrics['test_fgsm_delta_norm'] = metrics['test_fgsm_delta_norm'] / metrics['test_total']
    metrics['test_pgd10_loss'] = metrics['test_pgd10_loss'] / metrics['test_total']
    metrics['test_pgd10_correct'] = metrics['test_pgd10_correct'] / metrics['test_total']
    metrics['test_pgd10_delta_norm'] = metrics['test_pgd10_delta_norm'] / metrics['test_total']
    metrics['test_fgsm_20_loss'] = metrics['test_fgsm_20_loss'] / metrics['test_total']
    metrics['test_fgsm_20_correct'] = metrics['test_fgsm_20_correct'] / metrics['test_total']
    metrics['test_fgsm_60_loss'] = metrics['test_fgsm_60_loss'] / metrics['test_total']
    metrics['test_fgsm_60_correct'] = metrics['test_fgsm_60_correct'] / metrics['test_total']
    metrics['test1_w_clamp_loss'] = metrics['test1_w_clamp_loss'] / metrics['test_total']
    metrics['test1_w_clamp_correct'] = metrics['test1_w_clamp_correct'] / metrics['test_total']
    metrics['test1_wo_clamp_loss'] = metrics['test1_wo_clamp_loss'] / metrics['test_total']
    metrics['test1_wo_clamp_correct'] = metrics['test1_wo_clamp_correct'] / metrics['test_total']
    metrics['test2_loss'] = metrics['test2_loss'] / metrics['test_total']
    metrics['test2_correct'] = metrics['test2_correct'] / metrics['test_total']
    metrics['test_rs_w_clamp_loss'] = metrics['test_rs_w_clamp_loss'] / metrics['test_total']
    metrics['test_rs_w_clamp_correct'] = metrics['test_rs_w_clamp_correct'] / metrics['test_total']
    metrics['test_rs_wo_clamp_loss'] = metrics['test_rs_wo_clamp_loss'] / metrics['test_total']
    metrics['test_rs_wo_clamp_correct'] = metrics['test_rs_wo_clamp_correct'] / metrics['test_total']
    metrics['test_diff_rs_w_clamp_loss'] = metrics['test_diff_rs_w_clamp_loss'] / metrics['test_total']
    metrics['test_diff_rs_w_clamp_correct'] = metrics['test_diff_rs_w_clamp_correct'] / metrics['test_total']
    metrics['test_diff_rs_wo_clamp_loss'] = metrics['test_diff_rs_wo_clamp_loss'] / metrics['test_total']
    metrics['test_diff_rs_wo_clamp_correct'] = metrics['test_diff_rs_wo_clamp_correct'] / metrics['test_total']
    metrics['test_df50_loop'] = metrics['test_df50_loop'] / metrics['test_total']
    metrics['test_df50_perturbation_norm'] = metrics['test_df50_perturbation_norm'] / metrics['test_total']

    return metrics


def tb_writer(writer, epoch, lr, train_clean_loss, train_clean_acc, train_fgsm_loss, train_fgsm_acc, train_fgsm_delta_norm, train_fgsm_20_loss, train_fgsm_20_acc, train_fgsm_60_loss, train_fgsm_60_acc,
              metrics):
    writer.add_scalars('loss',
                       {'train_clean': train_clean_loss, 'train_fgsm': train_fgsm_loss, 'train_fgsm_20': train_fgsm_20_loss, 'train_fgsm_60': train_fgsm_60_loss,
                        'test_clean': metrics['test_clean_loss'], 'test_fgsm': metrics['test_fgsm_loss'], 'test_pgd': metrics['test_pgd10_loss'],
                        'test_fgsm_20': metrics['test_fgsm_20_loss'], 'test_fgsm_60': metrics['test_fgsm_60_loss'],
                        'test1_w_clamp': metrics['test1_w_clamp_loss'], 'test1_wo_clamp': metrics['test1_wo_clamp_loss'],
                        'test2': metrics['test2_loss'],
                        'test_rs_w_clamp': metrics['test_rs_w_clamp_loss'], 'test_rs_wo_clamp': metrics['test_rs_wo_clamp_loss'],
                        'test_diff_rs_w_clamp': metrics['test_diff_rs_w_clamp_loss'], 'test_diff_rs_wo_clamp': metrics['test_diff_rs_wo_clamp_loss']}, epoch)
    writer.add_scalars('accuracy',
                       {'train_clean': train_clean_acc, 'train_fgsm': train_fgsm_acc, 'train_fgsm_20': train_fgsm_20_acc, 'train_fgsm_60': train_fgsm_60_acc,
                        'test_clean': metrics['test_clean_correct'], 'test_fgsm': metrics['test_fgsm_correct'], 'test_pgd': metrics['test_pgd10_correct'],
                        'test_fgsm_20': metrics['test_fgsm_20_correct'], 'test_fgsm_60': metrics['test_fgsm_60_correct'],
                        'test1_w_clamp': metrics['test1_w_clamp_correct'], 'test1_wo_clamp': metrics['test1_wo_clamp_correct'],
                        'test2': metrics['test2_correct'],
                        'test_rs_w_clamp': metrics['test_rs_w_clamp_correct'], 'test_rs_wo_clamp': metrics['test_rs_wo_clamp_correct'],
                        'test_diff_rs_w_clamp': metrics['test_diff_rs_w_clamp_correct'], 'test_diff_rs_wo_clamp': metrics['test_diff_rs_wo_clamp_correct']}, epoch)
    writer.add_scalar('learning rate', lr, epoch)
    writer.add_scalar('test_input_grad_norm', metrics['test_input_grad_norm'], epoch)
    writer.add_scalar('test_df50_loop', metrics['test_df50_loop'], epoch)
    writer.add_scalars('delta_norm', {'train_fgsm': train_fgsm_delta_norm, 'test_fgsm': metrics['test_fgsm_delta_norm'], 'test_pgd10': metrics['test_pgd10_delta_norm'], 'test_df50': metrics['test_df50_perturbation_norm']}, epoch)


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
    # Model
    args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda'
        cudnn.benchmark = True
    logger.info(args)
    logger.info(f"model trained on {args.device}")

    torch.manual_seed(233)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    trainset = datasets.CIFAR10(
        root=args.dataset_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform_test)
    subsetA, _ = torch.utils.data.random_split(testset, [2000, 8000],
                                               generator=torch.Generator().manual_seed(6666666666))
    testloader = torch.utils.data.DataLoader(subsetA, batch_size=args.batch_size, shuffle=False)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if torch.cuda.device_count() > 1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    logger.info('==> Resuming from checkpoint..')
    assert os.path.isfile(os.path.join(CHECKPOINT_DIR,
                                       args.resumed_model_name)), f'Error: no asked checkpoint file {args.resumed_model_name} found! '
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name))
    model.load_state_dict(checkpoint['model'])

    args.epsilon = (args.epsilon / 255.) / std

    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Test Standard Loss \t Test Standard '
        'Acc \t Test Attack Loss \t Test Attack Acc \t Train fgsm norm \t Test pgd norm')

    global iteration
    iteration = 0
    for epoch in range(args.num_epochs):
        train(args, logger, writer, model, trainloader, testloader, optimizer, criterion)
    writer.close()


if __name__ == "__main__":
    main()
