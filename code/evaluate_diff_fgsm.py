"""Train CIFAR10 with PyTorch."""

import argparse
import logging
import os

from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18
from utils.deepfool import *


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset_path', default='./data', help='path of the dataset')
    parser.add_argument('--dataset', default='cifar10',  choices=['cifar10', 'svhn', 'cifar100'])

    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--batch_size', '-b', default=256, type=int)
    parser.add_argument('--train_fgsm_ratio', default=1, type=float)
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--resumed_model_name', default='standard_cifar.pth', help='the file name of resumed model')
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()


def eval(args, model, testloader, normalize):
    model.eval()
    metrics = {"test_clean_correct": 0,
               "test_fgsm10_wors_correct": 0,
               "test_fgsm20_wors_correct": 0,
               "test_fgsm30_wors_correct": 0,
               "test_fgsm40_wors_correct": 0,
               "test_fgsm50_wors_correct": 0,
               "test_fgsm60_wors_correct": 0,
               "test_fgsm70_wors_correct": 0,
               "test_fgsm80_wors_correct": 0,
               "test_fgsm90_wors_correct": 0,
               "test_fgsm100_wors_correct": 0,
               # "test_fgsm10_rs_correct": 0,
               # "test_fgsm20_rs_correct": 0,
               # "test_fgsm30_rs_correct": 0,
               # "test_fgsm40_rs_correct": 0,
               # "test_fgsm50_rs_correct": 0,
               # "test_fgsm60_rs_correct": 0,
               # "test_fgsm70_rs_correct": 0,
               # "test_fgsm80_rs_correct": 0,
               # "test_fgsm90_rs_correct": 0,
               # "test_fgsm100_rs_correct": 0,
               # "test_fgsm110_rs_correct": 0,
               # "test_fgsm120_rs_correct": 0,
               # "test_fgsm130_rs_correct": 0,
               # "test_fgsm140_rs_correct": 0,
               # "test_fgsm150_rs_correct": 0,
               "test_pgd10_correct": 0,
               "test_total": 0}
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        metrics['test_total'] += targets.size(0)
        # clean
        clean_outputs = model(normalize(inputs))
        metrics['test_clean_correct'] += (clean_outputs.max(dim=1)[1] == targets).sum().item()

        delta = torch.zeros_like(inputs).to(args.device)
        delta.requires_grad = True
        output = model(normalize(inputs + delta))
        loss = F.cross_entropy(output, targets)
        loss.backward()
        grad = delta.grad.detach()
        fgsm_delta_full = delta + args.train_fgsm_ratio * args.epsilon * torch.sign(grad)
        for i in range(10, 101, 10):
            fgsm_delta = i / 100. * fgsm_delta_full
            fgsm_delta = clamp(fgsm_delta, lower_limit - inputs, upper_limit - inputs).detach()
            fgsm_outputs = model(normalize(inputs + fgsm_delta))
            metrics["test_fgsm" + str(i) + "_wors_correct"] += (fgsm_outputs.max(dim=1)[1] == targets).sum().item()

        # loop, perturbation = deepfool_train(model, inputs, overshoot=0.02, max_iter=50, norm_dist='l_2',
        #                                     device=args.device, random_start=False, early_stop=True)
        # perturbation_norm = perturbation.view(perturbation.shape[0], -1).norm(dim=1)

        # _, perturbation_linf_ = deepfool_train(model, inputs, overshoot=0.02, max_iter=1, norm_dist='l_inf',
        #                                        device=args.device, random_start=False, early_stop=False)
        # perturbation_linf = clamp(perturbation_linf_, -args.epsilon, args.epsilon)
        # perturbation_linf = clamp(perturbation_linf, lower_limit - inputs, upper_limit - inputs).detach()
        # perturbation_linf_norm = perturbation_linf.view(perturbation_linf.shape[0], -1).norm(dim=1)

        # fgsm_delta = clamp(args.epsilon * torch.sign(grad), -args.epsilon, args.epsilon)
        # fgsm_delta = clamp(fgsm_delta, lower_limit - inputs, upper_limit - inputs).detach()
        # fgsm_delta_norm = fgsm_delta.view(fgsm_delta.shape[0], -1).norm(dim=1)
        #
        # cos_df50_fgsm = cal_cos_similarity(perturbation, fgsm_delta, perturbation_norm, fgsm_delta_norm)
        # cos_df50_df = cal_cos_similarity(perturbation, perturbation_linf, perturbation_norm, perturbation_linf_norm)
        # cos_fgsm_df = cal_cos_similarity(fgsm_delta, perturbation_linf, fgsm_delta_norm, perturbation_linf_norm)
        # metrics['test_cos_df50_fgsm'] += cos_df50_fgsm.sum().item()
        # metrics["test_cos_df50_df"] += cos_df50_df.sum().item()
        # metrics["test_cos_fgsm_df"] += cos_fgsm_df.sum().item()

        # delta = delta.detach()
        # for i in range(len(args.epsilon)):
        #     delta[:, i, :, :].uniform_(-args.epsilon[i][0][0].item(), args.epsilon[i][0][0].item())
        # delta = clamp(delta, lower_limit - inputs, upper_limit - inputs)
        # delta.requires_grad = True
        # output = model(inputs + delta)
        # loss = F.cross_entropy(output, targets)
        # loss.backward()
        # grad = delta.grad.detach()
        # grad_sign = torch.sign(grad)
        # for i in range(10, 151, 10):
        #     fgsm_delta = clamp(delta + i / 100. * args.epsilon * grad_sign, -args.epsilon, args.epsilon)
        #     fgsm_delta = clamp(fgsm_delta, lower_limit - inputs, upper_limit - inputs).detach()
        #     fgsm_outputs = model(inputs + fgsm_delta)
        #     metrics["test_fgsm" + str(i) + "_rs_correct"] += (fgsm_outputs.max(dim=1)[1] == targets).sum().item()

        pgd_delta = attack_pgd(model, inputs, targets, normalize, args.epsilon, 0.25 * args.epsilon, 10, 1, args.device, early_stop=True).detach()
        pgd_outputs = model(normalize(inputs + pgd_delta))
        metrics["test_pgd10_correct"] += (pgd_outputs.max(dim=1)[1] == targets).sum().item()

    metrics["test_clean_correct"] = 100 * metrics["test_clean_correct"] / metrics["test_total"]
    for i in range(10, 101, 10):
        metrics["test_fgsm" + str(i) + "_wors_correct"] = 100 * metrics["test_fgsm" + str(i) + "_wors_correct"] / metrics["test_total"]
    # for i in range(10, 151, 10):
    #     metrics["test_fgsm" + str(i) + "_rs_correct"] = metrics["test_fgsm" + str(i) + "_rs_correct"] / metrics["test_total"]
    metrics["test_pgd10_correct"] = 100 * metrics["test_pgd10_correct"] / metrics["test_total"]
    # metrics['test_cos_df50_fgsm'] = metrics['test_cos_df50_fgsm'] / metrics["test_total"]
    # metrics["test_cos_df50_df"] = metrics["test_cos_df50_df"] / metrics["test_total"]
    # metrics["test_cos_fgsm_df"] = metrics["test_cos_fgsm_df"] / metrics["test_total"]
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
    ])

    testset = datasets.CIFAR10(
        root=args.dataset_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    normalize = Normalize(cifar10_mu, cifar10_std)

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
    args.epsilon = args.epsilon / 255.

    logger.info('epoch \t clean \t wors 10-100 \t pgd10')

    for epoch in range(1, 201, args.interval):
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name + f'_{epoch}.pth'))
        model.load_state_dict(checkpoint['model'])
        metrics = eval(args, model, testloader, normalize)
        logger.info('%d %.2f  %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f  %.2f',
                    checkpoint['epoch'],
                    metrics["test_clean_correct"],
                    metrics["test_fgsm10_wors_correct"],
                    metrics["test_fgsm20_wors_correct"],
                    metrics["test_fgsm30_wors_correct"],
                    metrics["test_fgsm40_wors_correct"],
                    metrics["test_fgsm50_wors_correct"],
                    metrics["test_fgsm60_wors_correct"],
                    metrics["test_fgsm70_wors_correct"],
                    metrics["test_fgsm80_wors_correct"],
                    metrics["test_fgsm90_wors_correct"],
                    metrics["test_fgsm100_wors_correct"],
                    # metrics["test_fgsm10_rs_correct"],
                    # metrics["test_fgsm20_rs_correct"],
                    # metrics["test_fgsm30_rs_correct"],
                    # metrics["test_fgsm40_rs_correct"],
                    # metrics["test_fgsm50_rs_correct"],
                    # metrics["test_fgsm60_rs_correct"],
                    # metrics["test_fgsm70_rs_correct"],
                    # metrics["test_fgsm80_rs_correct"],
                    # metrics["test_fgsm90_rs_correct"],
                    # metrics["test_fgsm100_rs_correct"],
                    # metrics["test_fgsm110_rs_correct"],
                    # metrics["test_fgsm120_rs_correct"],
                    # metrics["test_fgsm130_rs_correct"],
                    # metrics["test_fgsm140_rs_correct"],
                    # metrics["test_fgsm150_rs_correct"],
                    metrics["test_pgd10_correct"])
                    # metrics['test_cos_df50_fgsm'],
                    # metrics["test_cos_df50_df"],
                    # metrics["test_cos_fgsm_df"])

if __name__ == "__main__":
    main()
