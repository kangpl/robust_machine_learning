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

    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--batch_size', '-b', default=256, type=int)
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--train_overshoot', default=0.02, type=float)
    parser.add_argument('--train_deepfool_norm_dist', default='l_inf', type=str)
    parser.add_argument('--resumed_model_name', default='standard_cifar.pth', help='the file name of resumed model')
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()


def eval(args, model, testloader):
    model.eval()
    metrics = {"test_clean_correct": 0,
               "test_df10_wors_correct": 0,
               "test_df20_wors_correct": 0,
               "test_df30_wors_correct": 0,
               "test_df40_wors_correct": 0,
               "test_df50_wors_correct": 0,
               "test_df60_wors_correct": 0,
               "test_df70_wors_correct": 0,
               "test_df80_wors_correct": 0,
               "test_df90_wors_correct": 0,
               "test_df100_wors_correct": 0,
               "test_pgd10_correct": 0,
               "test_total": 0}
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        metrics['test_total'] += targets.size(0)
        # clean
        clean_outputs = model(inputs)
        metrics['test_clean_correct'] += (clean_outputs.max(dim=1)[1] == targets).sum().item()

        _, perturbation = deepfool_train(model, inputs, overshoot=args.train_overshoot,
                                         max_iter=1,
                                         norm_dist=args.train_deepfool_norm_dist, device=args.device,
                                         random_start=False, early_stop=False)
        perturbation = clamp(args.epsilon * torch.sign(perturbation), -args.epsilon, args.epsilon)
        perturbation = clamp(perturbation, lower_limit - inputs, upper_limit - inputs).detach()

        for i in range(10, 101, 10):
            df_perturbation = i / 100. * perturbation
            df_outputs = model(inputs + df_perturbation)
            metrics["test_df" + str(i) + "_wors_correct"] += (df_outputs.max(dim=1)[1] == targets).sum().item()

        pgd_delta = attack_pgd(model, inputs, targets, args.epsilon, 0.25 * args.epsilon, 10, 1, args.device, early_stop=True).detach()
        pgd_outputs = model(clamp(inputs + pgd_delta, lower_limit, upper_limit))
        metrics["test_pgd10_correct"] += (pgd_outputs.max(dim=1)[1] == targets).sum().item()

    metrics["test_clean_correct"] = metrics["test_clean_correct"] / metrics["test_total"]
    for i in range(10, 101, 10):
        metrics["test_df" + str(i) + "_wors_correct"] = metrics["test_df" + str(i) + "_wors_correct"] / metrics["test_total"]
    metrics["test_pgd10_correct"] = metrics["test_pgd10_correct"] / metrics["test_total"]

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
    args.epsilon = (args.epsilon / 255.) / std

    logger.info(
        'epoch \t clean \t wors 10-100 \t pgd10')

    for epoch in range(1, 201, args.interval):
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name + f'_{epoch}.pth'))
        model.load_state_dict(checkpoint['model'])
        metrics = eval(args, model, testloader)
        logger.info('%d %.2f  %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f  %.2f',
                    checkpoint['epoch'],
                    metrics["test_clean_correct"],
                    metrics["test_df10_wors_correct"],
                    metrics["test_df20_wors_correct"],
                    metrics["test_df30_wors_correct"],
                    metrics["test_df40_wors_correct"],
                    metrics["test_df50_wors_correct"],
                    metrics["test_df60_wors_correct"],
                    metrics["test_df70_wors_correct"],
                    metrics["test_df80_wors_correct"],
                    metrics["test_df90_wors_correct"],
                    metrics["test_df100_wors_correct"],
                    metrics["test_pgd10_correct"])

if __name__ == "__main__":
    main()
