"""Train CIFAR10 with PyTorch."""

import argparse
import logging
import os
import time

import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from deepfool import *
from models.preact_resnet import PreActResNet18
from models.resnet import ResNet18


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset_path', default='./data', help='path of the dataset')

    parser.add_argument('--model', '-m', default='PreActResNet18', type=str)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', default='multistep', choices=['multistep', 'constant', 'cyclic'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.3, type=float)
    parser.add_argument('--lr_change_epoch', nargs='+', default=[100, 150], type=int)
    parser.add_argument('--batch_size', '-b', default=256, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])

    parser.add_argument('--train_overshoot', default=0.02, type=float)
    parser.add_argument('--train_deepfool_max_iter', default=1, type=int)
    parser.add_argument('--train_deepfool_norm_dist', default='l_inf', type=str)
    parser.add_argument('--train_deepfool_rs', action='store_true')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--eval_pgd_ratio', default=0.25, type=float)
    parser.add_argument('--eval_pgd_attack_iters', default=10, type=int)
    parser.add_argument('--eval_pgd_restarts', default=1, type=int)

    parser.add_argument('--finetune', action='store_true', help='finetune the pre-trained model with adversarial '
                                                                'samples or regularization')
    parser.add_argument('--resumed_model_name', default='standard_cifar.pth', help='the file name of resumed model')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_epoch', action='store_true')
    parser.add_argument('--exp_name', default='standard_cifar', help='used as filename of saved model, '
                                                                     'tensorboard and log')
    return parser.parse_args()


# Training
def train(args, model, trainloader, normalize, optimizer, criterion, step_lr_scheduler):
    model.train()
    train_clean_loss, train_clean_correct, train_df_loss, train_df_correct, train_df_perturbation_norm, train_total = 0, 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        _, perturbation = deepfool_train(model, inputs, normalize, overshoot=args.train_overshoot,
                                         max_iter=args.train_deepfool_max_iter, norm_dist=args.train_deepfool_norm_dist,
                                         device=args.device, random_start=args.train_deepfool_rs, epsilon=args.epsilon,
                                         early_stop=False)
        perturbation = torch.clamp(perturbation, min=-args.epsilon, max=args.epsilon)
        perturbation = clamp(perturbation, lower_limit - inputs, upper_limit - inputs).detach()
        perturbation_norm_after_clamp = perturbation.view(perturbation.shape[0], -1).norm(dim=1)

        df_outputs = model(normalize(inputs + perturbation))
        df_loss = criterion(df_outputs, targets)
        optimizer.zero_grad()
        df_loss.backward()
        optimizer.step()

        clean_outputs = model(normalize(inputs))
        clean_loss = criterion(clean_outputs, targets)

        train_df_loss += df_loss.item() * targets.size(0)
        train_df_correct += (df_outputs.max(dim=1)[1] == targets).sum().item()
        train_clean_loss += clean_loss.item() * targets.size(0)
        train_clean_correct += (clean_outputs.max(dim=1)[1] == targets).sum().item()
        train_total += targets.size(0)
        train_df_perturbation_norm += perturbation_norm_after_clamp.sum().item()
        if args.lr_schedule == 'cyclic':
            step_lr_scheduler.step()
    return train_clean_loss / train_total, 100. * train_clean_correct / train_total, \
           train_df_loss / train_total, 100. * train_df_correct / train_total, train_df_perturbation_norm / train_total


def eval(args, model, testloader, normalize, criterion, finaleval=False):
    model.eval()
    test_clean_loss, test_clean_correct, test_df_loss, test_df_correct, test_df_perturbation_norm, test_pgd10_loss, test_pgd10_correct, test_pgd10_delta_norm, test_total = 0, 0, 0, 0, 0, 0, 0, 0, 0
    test_input_grad_norm, test_df50_loop, test_df50_perturbation_norm = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # clean
        input_grad, clean_outputs, clean_loss = get_input_grad_v2(model, normalize(inputs), targets)
        input_grad_norm = input_grad.view(input_grad.shape[0], -1).norm(dim=1)

        _, perturbation = deepfool_train(model, inputs, normalize, overshoot=args.train_overshoot,
                                         max_iter=args.train_deepfool_max_iter, norm_dist=args.train_deepfool_norm_dist,
                                         device=args.device, random_start=args.train_deepfool_rs, epsilon=args.epsilon,
                                         early_stop=False)
        perturbation = torch.clamp(perturbation, min=-args.epsilon, max=args.epsilon)
        perturbation = clamp(perturbation, lower_limit - inputs, upper_limit - inputs).detach()
        perturbation_norm_after_clamp = perturbation.view(perturbation.shape[0], -1).norm(dim=1)
        df_outputs = model(normalize(inputs + perturbation))
        df_loss = criterion(df_outputs, targets)

        # pgd
        if finaleval:
            pgd_delta = attack_pgd(model, inputs, targets, normalize, args.epsilon, args.eval_pgd_ratio * args.epsilon,
                                   50, 10, args.device, early_stop=True).detach()
        else:
            pgd_delta = attack_pgd(model, inputs, targets, normalize, args.epsilon, args.eval_pgd_ratio * args.epsilon,
                                   args.eval_pgd_attack_iters, args.eval_pgd_restarts, args.device,
                                   early_stop=True).detach()
        pgd_outputs = model(normalize(inputs + pgd_delta))
        pgd_loss = criterion(pgd_outputs, targets)
        pgd_delta_norm = pgd_delta.view(pgd_delta.shape[0], -1).norm(dim=1)

        # deepfool
        loop, perturbation = deepfool_train(model, inputs, normalize, overshoot=0.02, max_iter=50, norm_dist='l_2',
                                            device=args.device, random_start=False, early_stop=True)
        perturbation_norm = perturbation.view(perturbation.shape[0], -1).norm(dim=1)

        test_clean_loss += clean_loss.item() * targets.size(0)
        test_clean_correct += (clean_outputs.max(dim=1)[1] == targets).sum().item()
        test_pgd10_loss += pgd_loss.item() * targets.size(0)
        test_pgd10_correct += (pgd_outputs.max(dim=1)[1] == targets).sum().item()
        test_pgd10_delta_norm += pgd_delta_norm.sum().item()
        test_total += targets.size(0)
        test_input_grad_norm += input_grad_norm.sum().item()
        test_df50_loop += loop.sum().item()
        test_df50_perturbation_norm += perturbation_norm.sum().item()
        test_df_loss += df_loss.item() * targets.size(0)
        test_df_correct += (df_outputs.max(dim=1)[1] == targets).sum().item()
        test_df_perturbation_norm += perturbation_norm_after_clamp.sum().item()

    return test_clean_loss / test_total, 100. * test_clean_correct / test_total, \
           test_df_loss / test_total, 100. * test_df_correct / test_total, test_df_perturbation_norm / test_total, \
           test_pgd10_loss / test_total, 100. * test_pgd10_correct / test_total, test_pgd10_delta_norm / test_total, \
           test_input_grad_norm / test_total, test_df50_loop / test_total, test_df50_perturbation_norm / test_total


def tb_writer(writer, epoch, lr, train_clean_loss, train_clean_acc, train_df_loss, train_df_acc,
              train_df_perturbation_norm,
              test_clean_loss, test_clean_acc, test_df_loss, test_df_acc, test_df_perturbation_norm, test_pgd10_loss,
              test_pgd10_acc, test_pgd10_delta_norm, test_input_grad_norm, test_df50_loop, test_df50_perturbation_norm):
    writer.add_scalars('loss',
                       {'train_clean': train_clean_loss, 'train_df': train_df_loss,
                        'test_clean': test_clean_loss, 'test_pgd': test_pgd10_loss,
                        'test_df': test_df_loss}, epoch)
    writer.add_scalars('accuracy',
                       {'train_clean': train_clean_acc, 'train_df': train_df_acc,
                        'test_clean': test_clean_acc, 'test_pgd': test_pgd10_acc,
                        'test_df': test_df_acc}, epoch)
    writer.add_scalar('learning rate', lr, epoch)
    writer.add_scalar('test_input_grad_norm', test_input_grad_norm, epoch)
    writer.add_scalar('test_df50_loop', test_df50_loop, epoch)
    writer.add_scalars('delta_norm', {'train_df': train_df_perturbation_norm, 'test_pgd10': test_pgd10_delta_norm,
                                      'test_df': test_df_perturbation_norm, 'test_df50': test_df50_perturbation_norm},
                       epoch)


def eval_init(args, writer, logger, model, trainloader, testloader, normalize, criterion, opt):
    model.eval()
    train_clean_loss, train_clean_correct, train_df_loss, train_df_correct, train_df_perturbation_norm, train_total = 0, 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        _, perturbation = deepfool_train(model, inputs, normalize, overshoot=args.train_overshoot,
                                         max_iter=args.train_deepfool_max_iter, norm_dist=args.train_deepfool_norm_dist,
                                         device=args.device, random_start=args.train_deepfool_rs, epsilon=args.epsilon,
                                         early_stop=False)
        perturbation = torch.clamp(perturbation, min=-args.epsilon, max=args.epsilon)
        perturbation = clamp(perturbation, lower_limit - inputs, upper_limit - inputs).detach()
        perturbation_norm_after_clamp = perturbation.view(perturbation.shape[0], -1).norm(dim=1)

        df_outputs = model(normalize(inputs + perturbation))
        df_loss = criterion(df_outputs, targets)

        clean_outputs = model(normalize(inputs))
        clean_loss = criterion(clean_outputs, targets)

        train_df_loss += df_loss.item() * targets.size(0)
        train_df_correct += (df_outputs.max(dim=1)[1] == targets).sum().item()
        train_clean_loss += clean_loss.item() * targets.size(0)
        train_clean_correct += (clean_outputs.max(dim=1)[1] == targets).sum().item()
        train_total += targets.size(0)
        train_df_perturbation_norm += perturbation_norm_after_clamp.sum().item()

    test_clean_loss, test_clean_acc, test_df_loss, test_df_acc, test_df_perturbation_norm, test_pgd_loss, test_pgd_acc, test_pgd10_delta_norm, test_input_grad_norm, test_df50_loop, test_df50_perturbation_norm = eval(
        args, model, testloader, normalize, criterion)
    tb_writer(writer, 0, opt.param_groups[0]['lr'],
              train_clean_loss / train_total, 100. * train_clean_correct / train_total,
              train_df_loss / train_total, 100. * train_df_correct / train_total,
              train_df_perturbation_norm / train_total,
              test_clean_loss, test_clean_acc, test_df_loss, test_df_acc, test_df_perturbation_norm, test_pgd_loss,
              test_pgd_acc, test_pgd10_delta_norm, test_input_grad_norm, test_df50_loop, test_df50_perturbation_norm)
    logger.info(
        '%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.2f \t \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f \t %.3f \t %.3f',
        0, -1, -1, opt.param_groups[0]['lr'], train_df_loss / train_total, 100. * train_df_correct / train_total,
        test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc, train_df_perturbation_norm / train_total,
        test_pgd10_delta_norm)


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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

    trainloader, testloader = get_loader(args, logger)
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

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError

    lr_steps = args.num_epochs * len(trainloader)
    if args.lr_schedule == 'cyclic':
        step_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
                                                              step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        step_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.lr_change_epoch,
                                                           gamma=0.1)

    args.epsilon = args.epsilon / 255.
    if args.finetune:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(os.path.join(CHECKPOINT_DIR,
                                           args.resumed_model_name)), f'Error: no asked checkpoint file {args.resumed_model_name} found! '
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.resumed_model_name))
        model.load_state_dict(checkpoint['model'])

    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Test Standard Loss \t Test Standard '
        'Acc \t Test Attack Loss \t Test Attack Acc \t Train df norm \t Test pgd norm')
    eval_init(args, writer, logger, model, trainloader, testloader, normalize, criterion, optimizer)

    best_test_pgd_acc = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_clean_loss, train_clean_acc, train_df_loss, train_df_acc, train_df_perturbation_norm = train(
            args, model, trainloader, normalize, optimizer, criterion, step_lr_scheduler)
        train_time = time.time()

        test_clean_loss, test_clean_acc, test_df_loss, test_df_acc, test_df_perturbation_norm, test_pgd_loss, test_pgd_acc, test_pgd10_delta_norm, test_input_grad_norm, test_df50_loop, test_df50_perturbation_norm = eval(
            args, model, testloader, normalize, criterion)
        test_time = time.time()

        tb_writer(writer, epoch + 1, optimizer.param_groups[0]['lr'],
                  train_clean_loss, train_clean_acc, train_df_loss, train_df_acc, train_df_perturbation_norm,
                  test_clean_loss, test_clean_acc, test_df_loss, test_df_acc, test_df_perturbation_norm, test_pgd_loss,
                  test_pgd_acc, test_pgd10_delta_norm, test_input_grad_norm, test_df50_loop,
                  test_df50_perturbation_norm)
        logger.info(
            '%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.2f \t \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f \t %.3f \t %.3f',
            epoch + 1, train_time - start_time, test_time - train_time, optimizer.param_groups[0]['lr'],
            train_df_loss, train_df_acc, test_clean_loss, test_clean_acc, test_pgd_loss, test_pgd_acc,
            train_df_perturbation_norm, test_pgd10_delta_norm)

        if args.lr_schedule == 'multistep':
            step_lr_scheduler.step()
        if args.save_epoch and epoch % 1 == 0:
            save_checkpoint(model, epoch + 1, train_df_loss, train_df_acc, test_clean_loss, test_clean_acc,
                            test_pgd_loss, test_pgd_acc,
                            os.path.join(CHECKPOINT_DIR, args.exp_name + f'_{epoch + 1}.pth'))
        if test_pgd_acc >= best_test_pgd_acc:
            save_checkpoint(model, epoch + 1, train_df_loss, train_df_acc, test_clean_loss, test_clean_acc,
                            test_pgd_loss, test_pgd_acc, os.path.join(CHECKPOINT_DIR, args.exp_name + f'_best.pth'))
            best_test_pgd_acc = test_pgd_acc

    save_checkpoint(model, epoch + 1, train_df_loss, train_df_acc, test_clean_loss, test_clean_acc, test_pgd_loss,
                    test_pgd_acc, os.path.join(CHECKPOINT_DIR, args.exp_name + f'_final.pth'))

    # Evaluate best and final
    logger.info('best')
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.exp_name + f'_best.pth'))
    model.load_state_dict(checkpoint['model'])
    best_clean_loss, best_clean_acc, _, _, _, best_pgd_loss, best_pgd_acc, _, _, _, _ = eval(args, model, testloader,
                                                                                             normalize, criterion,
                                                                                             finaleval=True)
    logger.info('%d \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f', checkpoint['epoch'], best_clean_loss, best_clean_acc,
                best_pgd_loss, best_pgd_acc)

    logger.info('final')
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, args.exp_name + f'_final.pth'))
    model.load_state_dict(checkpoint['model'])
    final_clean_loss, final_clean_acc, _, _, _, final_pgd_loss, final_pgd_acc, _, _, _, _ = eval(args, model,
                                                                                                 testloader, normalize,
                                                                                                 criterion,
                                                                                                 finaleval=True)
    logger.info('%d \t %.4f \t \t %.2f \t \t \t %.4f \t \t %.2f', checkpoint['epoch'], final_clean_loss,
                final_clean_acc, final_pgd_loss, final_pgd_acc)
    writer.close()


if __name__ == "__main__":
    main()
