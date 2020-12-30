"""Train CIFAR10 with PyTorch."""

import argparse
import logging
import math
import os

from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from deepfool import *
from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset_path', default='./data', help='path of the dataset')

    parser.add_argument('--epochs', nargs='+', default=[196, 16, 11, 6, 1], type=int)
    parser.add_argument('--num', default=1, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--train_overshoot', default=0.02, type=float)
    parser.add_argument('--train_fgsm_ratio', default=1, type=float)
    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--resumed_model_name', default='1201_fgsm_wors_{8}_1', help='the file name of resumed model')
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()


def draw_decision_boundary(logger, args, model, inputs, targets, e1_grad, e1_grad_norm, e2_grad, e2_grad_norm, palette, ax, label1,
                           label2, loop=-1, move=False):
    cos = cal_cos_similarity(e1_grad, e2_grad, e1_grad_norm, e2_grad_norm)
    angle = math.acos(cos)
    if math.tan(angle) == 0 or math.sin(angle) == 0:
        return
    matrix = np.array([[1, -1 / math.tan(angle)], [0, 1 / math.sin(angle)]])
    r_matrix = np.linalg.inv(matrix)
    e1 = e1_grad / e1_grad_norm
    e2 = e2_grad / e2_grad_norm

    logger.info(
        f'e1_grad_norm {round(e1_grad_norm.item(), 2)}, e2_grad_norm {round(e2_grad_norm.item(), 2)}, angle {round(angle / math.pi * 180, 2)}, cos {round(cos.item(), 2)}')
    x_indices = np.linspace(-5, 15, 200)
    y_indices = np.linspace(-5, 15, 200)
    x_array = np.outer(np.linspace(-5, 15, 200), np.ones(200))
    y_array = x_array.copy().T
    # loss_list = []
    label_list = []
    for x_ind in x_indices:
        perturbed_intputs = []
        for y_ind in y_indices:
            # from cartesian coordinate to [e1, e2] non-orthogonal coordinate
            new_index = np.matmul(matrix, [x_ind, y_ind])
            new_inputs = inputs + new_index[0] * e1 + new_index[1] * e2
            perturbed_intputs.append(new_inputs)

        perturbed_intputs_cat = torch.cat(perturbed_intputs, dim=0)
        perturbed_outputs = model(perturbed_intputs_cat)

        # t = torch.zeros(perturbed_intputs_cat.shape[0], dtype=torch.long).fill_(targets.item()).to(args.device)
        # perturbed_loss = F.cross_entropy(perturbed_outputs, t, reduction='none').detach()
        # loss_list.append(perturbed_loss.cpu().numpy()[None, :])
        label_list.append(perturbed_outputs.max(dim=1)[1].cpu().numpy()[None, :])
    #         print(label_list)
    # loss_array = np.concatenate(loss_list)
    label_array = np.concatenate(label_list)
    colors = np.empty(label_array.shape, dtype='<U7')
    for ix, iy in np.ndindex(label_array.shape):
        colors[ix, iy] = palette[label_array[ix, iy]]
    #         print(label_array[ix,iy], colors[ix,iy])

    e1_xy = np.matmul(r_matrix, np.array([e1_grad_norm.item(), 0]))
    e2_xy = np.matmul(r_matrix, np.array([0, e2_grad_norm.item()]))
    points = np.concatenate((e1_xy[None, :], e2_xy[None, :]))
    if not move:
        x_e1 = np.linspace(0, e1_xy[0], 100)
        y_e1 = np.linspace(0, e1_xy[1], 100)
    else:
        x_e1 = np.linspace(0, e1_xy[0], 100) + e2_xy[0]
        y_e1 = np.linspace(0, e1_xy[1], 100) + e2_xy[1]
    x_e2 = np.linspace(0, e2_xy[0], 100)
    y_e2 = np.linspace(0, e2_xy[1], 100)

    ax.scatter(x_array.flatten(), y_array.flatten(), s=1, c=colors.flatten(), marker=',')
    ax.plot(x_e1, y_e1, 'black')
    ax.plot(x_e2, y_e2, 'black')
    #     ax.scatter(points[:,0], points[:,1],  s=66)
    if loop == -1:
        label1 = label1 + f' L({round(e1_grad_norm.item(), 2)})'
    else:
        label1 = label1 + f'({loop}) L({round(e1_grad_norm.item(), 2)})'
    for i, txt in enumerate([label1, label2 + f' L({round(e2_grad_norm.item(), 2)}) cos({round(cos.item(), 2)})']):
        if i == 0 and move:
            text = ax.annotate(txt, (points[i, 0] + points[1, 0], points[i, 1] + points[1, 1]))
        if i == 1 and move:
            text = ax.annotate(txt, (0, 0))
        if not move:
            if i == 1:
                text = ax.annotate(txt, (points[i, 0], points[i, 1]))
            else:
                text = ax.annotate(txt, (points[i, 0], points[i, 1]))
        text.set_fontsize(10)


def draw_differ_epoch(logger, args, model, testloaderAll, testloader, epoch, label='test'):
    resumed_model_name = args.resumed_model_name + f'_{epoch}.pth'
    logger.info('======================' + resumed_model_name + '======================')
    checkpoint = torch.load(os.path.join(args.CHECKPOINT_DIR, resumed_model_name))
    logger.info('test clean acc ', checkpoint['test_standard_acc'], 'test pgd10 acc', checkpoint['test_attack_acc'])
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(testloaderAll):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = model(inputs)

        correct = (outputs.max(1)[1] == targets).sum().item()
        logger.info('clean_correct ', correct)

        loop50, perturbation50 = deepfool_train(model, inputs, overshoot=0.02, max_iter=50, norm_dist='l_2',
                                                device=args.device, random_start=False, early_stop=True)
        df50_outputs = model(inputs + perturbation50)
        df50_correct = (df50_outputs.max(1)[1] == targets).sum().item()
        df50_loop = loop50.mean().item()
        logger.info('df50_correct ', df50_correct, ' df50_loop ', df50_loop)

        pgd_delta = attack_pgd(model, inputs, targets, args.epsilon, 0.25 * args.epsilon, 10, 1, args.device, early_stop=True).detach()
        pgd_outputs = model(inputs + pgd_delta)
        pgd_correct = (pgd_outputs.max(1)[1] == targets).sum().item()
        logger.info('pgd10_correct', pgd_correct)

        pgd50_1_delta = attack_pgd(model, inputs, targets, args.epsilon, 0.25 * args.epsilon, 50, 1, args.device,
                                   early_stop=True).detach()
        pgd50_1_outputs = model(inputs + pgd50_1_delta)
        pgd50_1_correct = (pgd50_1_outputs.max(1)[1] == targets).sum().item()
        logger.info('pgd50_1_correct', pgd50_1_correct)

        pgd50_delta = attack_pgd(model, inputs, targets, args.epsilon, 0.25 * args.epsilon, 50, 10, args.device,
                                 early_stop=True).detach()
        pgd50_outputs = model(inputs + pgd50_delta)
        pgd50_correct = (pgd50_outputs.max(1)[1] == targets).sum().item()
        logger.info('pgd50_correct', pgd50_correct)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx < args.num:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            delta = torch.zeros_like(inputs).to(args.device)
            delta.requires_grad = True
            output = model(inputs + delta)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            grad = delta.grad.detach()
            fgsm_delta = clamp(args.epsilon * torch.sign(grad), -args.epsilon, args.epsilon)
            fgsm_delta = clamp(fgsm_delta, lower_limit - inputs, upper_limit - inputs).detach()
            fgsm_delta_norm = fgsm_delta.view(fgsm_delta.shape[0], -1).norm(dim=1)

            _, perturbation_linf_ = deepfool_train(model, inputs, overshoot=args.train_overshoot, max_iter=1,
                                                  norm_dist='l_inf', device=args.device, random_start=False,
                                                  early_stop=False)
            perturbation_linf = clamp(perturbation_linf_, -args.epsilon, args.epsilon)
            perturbation_linf = clamp(perturbation_linf, lower_limit - inputs, upper_limit - inputs).detach()
            perturbation_linf_norm = perturbation_linf.view(perturbation_linf.shape[0], -1).norm(dim=1)

            # _, perturbation_l2_ = deepfool_train(model, inputs, overshoot=args.train_overshoot, max_iter=1, norm_dist='l_2',
            #                                     device=args.device, random_start=False, early_stop=False)
            # perturbation_l2 = clamp(perturbation_l2_, -args.epsilon, args.epsilon)
            # perturbation_l2 = clamp(perturbation_l2, lower_limit - inputs, upper_limit - inputs).detach()
            # perturbation_l2_norm = perturbation_l2.view(perturbation_l2.shape[0], -1).norm(dim=1)

            # Dfgsm_Llinf_delta = clamp(torch.mul(abs(perturbation_linf_), torch.sign(grad)), -args.train_fgsm_ratio * args.epsilon, args.train_fgsm_ratio * args.epsilon)
            # Dfgsm_Llinf_delta = clamp(Dfgsm_Llinf_delta, lower_limit - inputs, upper_limit - inputs).detach()
            # Dfgsm_Llinf_delta_norm = Dfgsm_Llinf_delta.view(Dfgsm_Llinf_delta.shape[0], -1).norm(dim=1)
            #
            # Dfgsm_L2_delta = torch.mul(abs(perturbation_l2), torch.sign(grad))
            # Dfgsm_L2_delta = clamp(Dfgsm_L2_delta, lower_limit - inputs, upper_limit - inputs).detach()
            # Dfgsm_L2_delta_norm = Dfgsm_L2_delta.view(Dfgsm_L2_delta.shape[0], -1).norm(dim=1)
            #
            # Dlinf_Lfgsm_delta = clamp(args.train_fgsm_ratio * args.epsilon * torch.sign(perturbation_linf_), -args.epsilon, args.epsilon)
            # Dlinf_Lfgsm_delta = clamp(Dlinf_Lfgsm_delta, lower_limit - inputs, upper_limit - inputs).detach()
            # Dlinf_Lfgsm_delta_norm = Dlinf_Lfgsm_delta.view(Dlinf_Lfgsm_delta.shape[0], -1).norm(dim=1)
            #
            # Dl2_Lfgsm_delta = torch.mul(abs(fgsm_delta), torch.sign(perturbation_l2_))
            # Dl2_Lfgsm_delta = clamp(Dl2_Lfgsm_delta, lower_limit - inputs, upper_limit - inputs).detach()
            # Dl2_Lfgsm_delta_norm = Dl2_Lfgsm_delta.view(Dl2_Lfgsm_delta.shape[0], -1).norm(dim=1)

            loop, perturbation50 = deepfool_train(model, inputs, overshoot=0.02, max_iter=50, norm_dist='l_2',
                                                  device=args.device, random_start=False, early_stop=True)
            perturbation50_norm = perturbation50.view(perturbation50.shape[0], -1).norm(dim=1)
            logger.info('index', batch_idx, 'loop', loop.item())

            pgd10_delta = attack_pgd(model, inputs, targets, args.epsilon, 0.25 * args.epsilon, 10, 1, args.device,
                                     early_stop=True).detach()
            pgd10_delta_norm = pgd10_delta.view(pgd10_delta.shape[0], -1).norm(dim=1)
            pgd50_delta = attack_pgd(model, inputs, targets, args.epsilon, 0.25 * args.epsilon, 50, 10, args.device,
                                     early_stop=True).detach()
            pgd50_delta_norm = pgd50_delta.view(pgd50_delta.shape[0], -1).norm(dim=1)

            delta = delta.detach()
            for i in range(len(args.epsilon)):
                delta[:, i, :, :].uniform_(-args.epsilon[i][0][0].item(), args.epsilon[i][0][0].item())
            delta = clamp(delta, lower_limit - inputs, upper_limit - inputs)
            delta.requires_grad = True
            delta_norm = delta.view(delta.shape[0], -1).norm(dim=1)

            output = model(inputs + delta)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            grad = delta.grad.detach()
            grad_sign = torch.sign(grad)
            fgsm_dir = args.train_fgsm_ratio * args.epsilon * grad_sign
            fgsm_dir_norm = fgsm_dir.view(fgsm_dir.shape[0], -1).norm(dim=1)

            fgsm8_delta = clamp(delta + fgsm_dir, -args.epsilon, args.epsilon)
            fgsm8_delta = clamp(fgsm8_delta, lower_limit - inputs, upper_limit - inputs).detach()
            fgsm8_delta_norm = fgsm8_delta.view(fgsm8_delta.shape[0], -1).norm(dim=1)
            fgsm4_delta = clamp(delta + 0.5 * fgsm_dir / args.train_fgsm_ratio, -args.epsilon, args.epsilon)
            fgsm4_delta = clamp(fgsm4_delta, lower_limit - inputs, upper_limit - inputs).detach()
            fgsm4_delta_norm = fgsm4_delta.view(fgsm4_delta.shape[0], -1).norm(dim=1)

            # _, linf_delta = deepfool_eval_without_rs(model, (inputs + delta).detach(), overshoot=args.train_overshoot,
            #                                          norm_dist='l_inf', device=args.device)
            # linf_delta_norm = linf_delta.view(linf_delta.shape[0], -1).norm(dim=1)
            # _, l2_delta = deepfool_eval_without_rs(model, (inputs + delta).detach(), overshoot=args.train_overshoot,
            #                                        norm_dist='l_2', device=args.device)
            # l2_delta_norm = l2_delta.view(l2_delta.shape[0], -1).norm(dim=1)

            # linf_rs_delta = clamp(delta + linf_delta, -args.epsilon, args.epsilon)
            # linf_rs_delta = clamp(linf_rs_delta, lower_limit - inputs, upper_limit - inputs).detach()
            # linf_rs_delta_norm = linf_rs_delta.view(linf_rs_delta.shape[0], -1).norm(dim=1)
            # l2_rs_delta = clamp(delta + l2_delta, -args.epsilon, args.epsilon)
            # l2_rs_delta = clamp(l2_rs_delta, lower_limit - inputs, upper_limit - inputs).detach()
            # l2_rs_delta_norm = l2_rs_delta.view(l2_rs_delta.shape[0], -1).norm(dim=1)

            # _, Dl2_Llinf_delta = deepfool_Dl2_Llinf(model, inputs, overshoot=args.train_overshoot, device=args.device)
            # Dl2_Llinf_delta = clamp(Dl2_Llinf_delta, -args.epsilon, args.epsilon)
            # Dl2_Llinf_delta = clamp(Dl2_Llinf_delta, lower_limit - inputs, upper_limit - inputs).detach()
            # Dl2_Llinf_delta_norm = Dl2_Llinf_delta.view(Dl2_Llinf_delta.shape[0], -1).norm(dim=1)

            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19) = plt.subplots(1, 19, figsize=(95, 5))
            palette = {0: '#d71d1d', 1: '#e22828', 2: '#e43a3a', 3: '#e74b4b', 4: '#e95d5d', 5: '#eb6f6f', 6: '#ee8181',
                       7: '#f09393', 8: '#f3a5a5', 9: '#f5b7b7'}
            palette[targets.item()] = '#6ec26e'
            logger.info(f"between df50 and fgsm_wors{args.epsilon_num}")
            draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, fgsm_delta,
                                   fgsm_delta_norm, palette, ax1, 'df50', f'fgsm_wors{args.epsilon_num}', loop=int(loop.item()))
            logger.info(f"between df50 and fgsm_rs{args.epsilon_num * args.train_fgsm_ratio}")
            draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, fgsm8_delta,
                                   fgsm8_delta_norm, palette, ax3, 'df50', f'fgsm_rs{args.epsilon_num * args.train_fgsm_ratio}', loop=int(loop.item()))
            logger.info(f"between df50 and fgsm_rs{args.epsilon_num * 0.5}")
            draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, fgsm4_delta,
                                   fgsm4_delta_norm, palette, ax4, 'df50', f'fgsm_rs{args.epsilon_num * 0.5}', loop=int(loop.item()))
            logger.info(f"between df50 and delta")
            draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, delta, delta_norm,
                                   palette, ax5, 'df50', 'delta', loop=int(loop.item()))
            logger.info(f"between fgsm_dir and delta")
            draw_decision_boundary(logger, args, model, inputs, targets, fgsm_dir, fgsm_dir_norm, delta, delta_norm, palette, ax6,
                                   'fgsm_dir', 'delta', move=True)
            logger.info(f"between df50 and df_linf({args.train_overshoot})")
            draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, perturbation_linf,
                                   perturbation_linf_norm, palette, ax2, 'df50', f'df_linf({args.train_overshoot})',
                                   loop=int(loop.item()))
            # logger.info(f"between df50 and Dfgsm_Llinf")
            # draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm,
            #                        Dfgsm_Llinf_delta,
            #                        Dfgsm_Llinf_delta_norm, palette, ax6, 'df50', 'Dfgsm_Llinf',
            #                        loop=int(loop.item()))
            # logger.info(f"between df50 and Dlinf_Lfgsm")
            # draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm,
            #                        Dlinf_Lfgsm_delta,
            #                        Dlinf_Lfgsm_delta_norm, palette, ax7, 'df50', 'Dlinf_Lfgsm',
            #                        loop=int(loop.item()))
            # logger.info(f"between linf_dir({args.train_overshoot}) and delta")
            # draw_decision_boundary(logger, args, model, inputs, targets, linf_delta, linf_delta_norm, delta, delta_norm,
            #                        palette, ax8,
            #                        f'linf_dir({args.train_overshoot})', 'delta', move=True)
            # logger.info(f"between df50 and df_rs_linf({args.train_overshoot})")
            # draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm,
            #                        linf_rs_delta,
            #                        linf_rs_delta_norm, palette, ax9, 'df50', f'df_rs_linf({args.train_overshoot})',
            #                        loop=int(loop.item()))
            # if loop != 1:
            #     logger.info(f"between df50 and df_l2({args.train_overshoot})")
            #     draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, perturbation_l2,
            #                            perturbation_l2_norm, palette, ax11, 'df50', f'df_l2({args.train_overshoot})',
            #                            loop=int(loop.item()))
            # logger.info(f"between df50 and Dfgsm_L2")
            # draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm,
            #                        Dfgsm_L2_delta,
            #                        Dfgsm_L2_delta_norm, palette, ax12, 'df50', 'Dfgsm_L2',
            #                        loop=int(loop.item()))
            # logger.info(f"between df50 and Dl2_Lfgsm")
            # draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm,
            #                        Dl2_Lfgsm_delta,
            #                        Dl2_Lfgsm_delta_norm, palette, ax13, 'df50', 'Dl2_Lfgsm',
            #                        loop=int(loop.item()))
            # logger.info(f"between l2_dir({args.train_overshoot}) and delta")
            # draw_decision_boundary(logger, args, model, inputs, targets, l2_delta, l2_delta_norm, delta, delta_norm, palette, ax12,
            #                        f'l2_dir({args.train_overshoot})', 'delta', move=True)
            # logger.info(f"between df50 and df_rs_l2({args.train_overshoot})")
            # draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, l2_rs_delta,
            #                        l2_rs_delta_norm, palette, ax13, 'df50', f'df_rs_l2({args.train_overshoot})',
            #                        loop=int(loop.item()))
            # logger.info(f"between df50 and Dl2_Llinf({args.train_overshoot})")
            # draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, Dl2_Llinf_delta,
            #                        Dl2_Llinf_delta_norm, palette, ax16, 'df50', f'Dl2_Llinf({args.train_overshoot})',
            #                        loop=int(loop.item()))

            logger.info("between df50 and pgd10")
            draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, pgd10_delta,
                                   pgd10_delta_norm, palette, ax7, 'df50', 'pgd10', loop=int(loop.item()))
            logger.info("between df50 and pgd50")
            draw_decision_boundary(logger, args, model, inputs, targets, perturbation50, perturbation50_norm, pgd50_delta,
                                   pgd50_delta_norm, palette, ax8, 'df50', 'pgd50', loop=int(loop.item()))
            logger.info("between pgd10 and pgd50")
            draw_decision_boundary(logger, args, model, inputs, targets, pgd10_delta, pgd10_delta_norm, pgd50_delta, pgd50_delta_norm,
                                   palette, ax9, 'pgd10', 'pgd50')
            plt.savefig('./images/' + resumed_model_name + label + f'_f{batch_idx}.png')
            # plt.show()
            logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            plt.close()


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
    args.CHECKPOINT_DIR = CHECKPOINT_DIR
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
    subsetTrain, _ = torch.utils.data.random_split(trainset, [100, 49900],
                                                   generator=torch.Generator().manual_seed(6666666666))
    trainloader = torch.utils.data.DataLoader(subsetTrain, batch_size=1, shuffle=False)
    trainloaderAll = torch.utils.data.DataLoader(subsetTrain, batch_size=100, shuffle=False)

    testset = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform_test)
    subsetA, _ = torch.utils.data.random_split(testset, [100, 9900],
                                               generator=torch.Generator().manual_seed(6666666666))
    testloader = torch.utils.data.DataLoader(subsetA, batch_size=1, shuffle=False)
    testloaderAll = torch.utils.data.DataLoader(subsetA, batch_size=100, shuffle=False)

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
    args.epsilon_num = args.epsilon
    args.epsilon = (args.epsilon / 255.) / std

    for epoch in args.epochs:
        draw_differ_epoch(logger, args, model, testloaderAll, testloader, epoch, 'test')
        # draw_differ_epoch(logger, args, model, trainloaderAll, trainloader, epoch, 'train')


if __name__ == "__main__":
    main()
