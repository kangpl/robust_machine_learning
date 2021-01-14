"""Train CIFAR10 with PyTorch."""

import argparse
import logging
import os

from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18
from deepfool import *


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset_path', default='./data', help='path of the dataset')
    parser.add_argument('--type', default='test', type=str, help='type of the dataset, train or test')
    parser.add_argument('--mode', default='eval', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--batch_size', '-b', default=256, type=int)
    parser.add_argument('--train_fgsm_ratio', default=1, type=float)
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--resumed_model_name', default='standard_cifar.pth', help='the file name of resumed model')
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()


def eval(args, model, loader, type='test'):
    if args.mode == 'eval':
        model.eval()
    elif args.mode == 'train':
        model.train()
    else:
        raise NotImplementedError
    metrics = {f"{type}_wors_perturbed_loss": 0,
               f"{type}_wors_loss_diff": 0,
               f"{type}_wors_acc": 0,
               f"{type}_test1_perturbed_loss": 0,
               f"{type}_test1_loss_diff": 0,
               f"{type}_test1_acc": 0,
               f"{type}_test2_perturbed_loss": 0,
               f"{type}_test2_loss_diff": 0,
               f"{type}_test2_acc": 0,
               f"{type}_rs_perturbed_loss": 0,
               f"{type}_rs_loss_diff": 0,
               f"{type}_rs_acc": 0,
               f"{type}_rs_wo_clamp_perturbed_loss": 0,
               f"{type}_rs_wo_clamp_loss_diff": 0,
               f"{type}_rs_wo_clamp_acc": 0,
               f"{type}_rs_diff_perturbed_loss": 0,
               f"{type}_rs_diff_loss_diff": 0,
               f"{type}_rs_diff_acc": 0,
               f"{type}_rs_diff_wo_clamp_perturbed_loss": 0,
               f"{type}_rs_diff_wo_clamp_loss_diff": 0,
               f"{type}_rs_diff_wo_clamp_acc": 0,
               f"{type}_total": 0}
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        zero = torch.zeros_like(inputs).to(args.device)
        zero.requires_grad = True
        output_clean = model(inputs + zero)
        loss_clean = F.cross_entropy(output_clean, targets, reduction='none')
        loss_clean.mean().backward()
        grad_input = zero.grad.detach()
        fgsm_delta = clamp(args.train_fgsm_ratio * args.epsilon * torch.sign(grad_input), -args.epsilon, args.epsilon)
        fgsm_delta = clamp(fgsm_delta, lower_limit - inputs, upper_limit - inputs).detach()
        output_wors = model(inputs + fgsm_delta)
        loss_wors = F.cross_entropy(output_wors, targets, reduction='none')
        diff_wors = loss_wors - loss_clean
        metrics[f'{type}_wors_perturbed_loss'] += loss_wors.sum().item()
        metrics[f'{type}_wors_loss_diff'] += diff_wors.sum().item()
        metrics[f'{type}_wors_acc'] += (output_wors.max(dim=1)[1] == targets).sum().item()

        delta = zero.detach()
        for i in range(len(args.epsilon)):
            delta[:, i, :, :].uniform_(-args.epsilon[i][0][0].item(), args.epsilon[i][0][0].item())
        delta = clamp(delta, lower_limit - inputs, upper_limit - inputs)
        fgsm_test1 = clamp(delta + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_input), -args.epsilon, args.epsilon)
        fgsm_test1 = clamp(fgsm_test1, lower_limit - inputs, upper_limit - inputs).detach()
        output_test1 = model(inputs + fgsm_test1)
        loss_test1 = F.cross_entropy(output_test1, targets, reduction='none')
        diff_test1 = loss_test1 - loss_clean
        metrics[f'{type}_test1_perturbed_loss'] += loss_test1.sum().item()
        metrics[f'{type}_test1_loss_diff'] += diff_test1.sum().item()
        metrics[f'{type}_test1_acc'] += (output_test1.max(dim=1)[1] == targets).sum().item()

        delta.requires_grad = True
        output = model(inputs + delta)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        grad_random = delta.grad.detach()
        fgsm_test2 = clamp(args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random), -args.epsilon, args.epsilon)
        fgsm_test2 = clamp(fgsm_test2, lower_limit - inputs, upper_limit - inputs).detach()
        output_test2 = model(inputs + fgsm_test2)
        loss_test2 = F.cross_entropy(output_test2, targets, reduction='none')
        diff_test2 = loss_test2 - loss_clean
        metrics[f'{type}_test2_perturbed_loss'] += loss_test2.sum().item()
        metrics[f'{type}_test2_loss_diff'] += diff_test2.sum().item()
        metrics[f'{type}_test2_acc'] += (output_test2.max(dim=1)[1] == targets).sum().item()

        fgsm_rs = clamp(delta + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random), -args.epsilon, args.epsilon)
        fgsm_rs = clamp(fgsm_rs, lower_limit - inputs, upper_limit - inputs).detach()
        output_rs = model(inputs + fgsm_rs)
        loss_rs = F.cross_entropy(output_rs, targets, reduction='none')
        diff_rs = loss_rs - loss_clean
        metrics[f'{type}_rs_perturbed_loss'] += loss_rs.sum().item()
        metrics[f'{type}_rs_loss_diff'] += diff_rs.sum().item()
        metrics[f'{type}_rs_acc'] += (output_rs.max(dim=1)[1] == targets).sum().item()

        fgsm_rs_wo_clamp = delta + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random)
        fgsm_rs_wo_clamp = clamp(fgsm_rs_wo_clamp, lower_limit - inputs, upper_limit - inputs).detach()
        output_rs_wo_clamp = model(inputs + fgsm_rs_wo_clamp)
        loss_rs_wo_clamp = F.cross_entropy(output_rs_wo_clamp, targets, reduction='none')
        diff_rs_wo_clamp = loss_rs_wo_clamp - loss_clean
        metrics[f"{type}_rs_wo_clamp_perturbed_loss"] += loss_rs_wo_clamp.sum().item()
        metrics[f"{type}_rs_wo_clamp_loss_diff"] += diff_rs_wo_clamp.sum().item()
        metrics[f"{type}_rs_wo_clamp_acc"] += (output_rs_wo_clamp.max(dim=1)[1] == targets).sum().item()

        delta2 = torch.zeros_like(inputs).to(args.device)
        for i in range(len(args.epsilon)):
            delta2[:, i, :, :].uniform_(-args.epsilon[i][0][0].item(), args.epsilon[i][0][0].item())
        delta2 = clamp(delta2, lower_limit - inputs, upper_limit - inputs)
        fgsm_rs_diff = clamp(delta2 + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random), -args.epsilon, args.epsilon)
        fgsm_rs_diff = clamp(fgsm_rs_diff, lower_limit - inputs, upper_limit - inputs).detach()
        output_rs_diff = model(inputs + fgsm_rs_diff)
        loss_rs_diff = F.cross_entropy(output_rs_diff, targets, reduction='none')
        diff_rs_diff = loss_rs_diff - loss_clean
        metrics[f"{type}_rs_diff_perturbed_loss"] += loss_rs_diff.sum().item()
        metrics[f"{type}_rs_diff_loss_diff"] += diff_rs_diff.sum().item()
        metrics[f"{type}_rs_diff_acc"] += (output_rs_diff.max(dim=1)[1] == targets).sum().item()

        fgsm_rs_diff_wo_clamp = delta2 + args.train_fgsm_ratio * args.epsilon * torch.sign(grad_random)
        fgsm_rs_diff_wo_clamp = clamp(fgsm_rs_diff_wo_clamp, lower_limit - inputs, upper_limit - inputs).detach()
        output_rs_diff_wo_clamp = model(inputs + fgsm_rs_diff_wo_clamp)
        loss_rs_diff_wo_clamp = F.cross_entropy(output_rs_diff_wo_clamp, targets, reduction='none')
        diff_rs_diff_wo_clamp = loss_rs_diff_wo_clamp - loss_clean
        metrics[f"{type}_rs_diff_wo_clamp_perturbed_loss"] += loss_rs_diff_wo_clamp.sum().item()
        metrics[f"{type}_rs_diff_wo_clamp_loss_diff"] += diff_rs_diff_wo_clamp.sum().item()
        metrics[f"{type}_rs_diff_wo_clamp_acc"] += (output_rs_diff_wo_clamp.max(dim=1)[1] == targets).sum().item()

        metrics[f"{type}_total"] += targets.size(0)

    metrics[f'{type}_wors_perturbed_loss'] = metrics[f'{type}_wors_perturbed_loss'] / metrics[f"{type}_total"]
    metrics[f'{type}_wors_loss_diff'] = metrics[f'{type}_wors_loss_diff'] / metrics[f"{type}_total"]
    metrics[f'{type}_wors_acc'] = metrics[f'{type}_wors_acc'] / metrics[f"{type}_total"]
    metrics[f'{type}_test1_perturbed_loss'] = metrics[f'{type}_test1_perturbed_loss'] / metrics[f"{type}_total"]
    metrics[f'{type}_test1_loss_diff'] = metrics[f'{type}_test1_loss_diff'] / metrics[f"{type}_total"]
    metrics[f'{type}_test1_acc'] = metrics[f'{type}_test1_acc'] / metrics[f"{type}_total"]
    metrics[f'{type}_test2_perturbed_loss'] = metrics[f'{type}_test2_perturbed_loss'] / metrics[f"{type}_total"]
    metrics[f'{type}_test2_loss_diff'] =metrics[f'{type}_test2_loss_diff'] / metrics[f"{type}_total"]
    metrics[f'{type}_test2_acc'] = metrics[f'{type}_test2_acc'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_perturbed_loss'] = metrics[f'{type}_rs_perturbed_loss'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_loss_diff'] = metrics[f'{type}_rs_loss_diff'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_acc'] = metrics[f'{type}_rs_acc'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_wo_clamp_perturbed_loss'] = metrics[f'{type}_rs_wo_clamp_perturbed_loss'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_wo_clamp_loss_diff'] = metrics[f'{type}_rs_wo_clamp_loss_diff'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_wo_clamp_acc'] = metrics[f'{type}_rs_wo_clamp_acc'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_diff_perturbed_loss'] = metrics[f'{type}_rs_diff_perturbed_loss'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_diff_loss_diff'] = metrics[f'{type}_rs_diff_loss_diff'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_diff_acc'] = metrics[f'{type}_rs_diff_acc'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_diff_wo_clamp_perturbed_loss'] = metrics[f'{type}_rs_diff_wo_clamp_perturbed_loss'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_diff_wo_clamp_loss_diff'] = metrics[f'{type}_rs_diff_wo_clamp_loss_diff'] / metrics[f"{type}_total"]
    metrics[f'{type}_rs_diff_wo_clamp_acc'] = metrics[f'{type}_rs_diff_wo_clamp_acc'] / metrics[f"{type}_total"]
    return metrics


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
    trainset = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform_train)
    subsetTrain, _ = torch.utils.data.random_split(trainset, [10000, 40000],
                                                   generator=torch.Generator().manual_seed(6666666666))
    trainloader = torch.utils.data.DataLoader(subsetTrain, batch_size=args.batch_size, shuffle=False, num_workers=2)

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
    args.epsilon = (args.epsilon / 255.) / std

    logger.info('epoch \t perturbed loss \t loss diff \t acc ')

    for epoch in range(1, 201, args.interval):
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name + f'_{epoch}.pth'))
        model.load_state_dict(checkpoint['model'])
        if args.type == 'test':
            if epoch == 1:
                logger.info("=============================evaluate test dataset=============================")
            metrics = eval(args, model, testloader, 'test')
        elif args.type == 'train':
            if epoch == 1:
                logger.info("=============================evaluate train dataset=============================")
            metrics = eval(args, model, trainloader, 'train')
        else:
            raise NotImplementedError
        logger.info('%d  %.3f %.3f %.4f  %.3f %.3f %.4f  %.3f %.3f %.4f  %.3f %.3f %.4f  %.3f %.3f %.4f  %.3f %.3f %.4f  %.3f %.3f %.4f',
                    checkpoint['epoch'],
                    metrics[f'{args.type}_wors_perturbed_loss'],
                    metrics[f'{args.type}_wors_loss_diff'],
                    metrics[f'{args.type}_wors_acc'],
                    metrics[f'{args.type}_test1_perturbed_loss'],
                    metrics[f'{args.type}_test1_loss_diff'],
                    metrics[f'{args.type}_test1_acc'],
                    metrics[f'{args.type}_test2_perturbed_loss'],
                    metrics[f'{args.type}_test2_loss_diff'],
                    metrics[f'{args.type}_test2_acc'],
                    metrics[f'{args.type}_rs_perturbed_loss'],
                    metrics[f'{args.type}_rs_loss_diff'],
                    metrics[f'{args.type}_rs_acc'],
                    metrics[f'{args.type}_rs_wo_clamp_perturbed_loss'],
                    metrics[f'{args.type}_rs_wo_clamp_loss_diff'],
                    metrics[f'{args.type}_rs_wo_clamp_acc'],
                    metrics[f'{args.type}_rs_diff_perturbed_loss'],
                    metrics[f'{args.type}_rs_diff_loss_diff'],
                    metrics[f'{args.type}_rs_diff_acc'],
                    metrics[f'{args.type}_rs_diff_wo_clamp_perturbed_loss'],
                    metrics[f'{args.type}_rs_diff_wo_clamp_loss_diff'],
                    metrics[f'{args.type}_rs_diff_wo_clamp_acc']
                    )


if __name__ == "__main__":
    main()
