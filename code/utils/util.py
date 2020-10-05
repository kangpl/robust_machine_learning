"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""

import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_batch_l2_norm(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    return norms


def get_mean_and_std():
    """Compute the mean and std value of dataset."""
    # load data
    # load the training data
    train_data = datasets.CIFAR10('../data', train=True, download=True)
    # use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    # calculate the mean and std along the (0, 1) axes
    train_mean = np.mean(x, axis=(0, 1)) / 255
    train_std = np.std(x, axis=(0, 1)) / 255
    return train_mean, train_std


def get_loader(args, logger):
    # Data
    logger.info('==> Preparing data..')
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

    testset = datasets.CIFAR10(
        root=args.dataset_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, device, norm="l_inf", early_stop=False):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True

        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def get_input_grad(model, X, y, delta_init='none', backprop=False, device='cuda'):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True).to(device)
    else:
        raise ValueError('wrong delta init')

    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad = grad.detach()
    return grad


def get_input_grad_v2(model, X, y):
    X.requires_grad_()
    output = model(X)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, X, create_graph=False)[0]
    return grad.detach()


def save_checkpoint(model, epoch, train_loss, train_acc, test_standard_loss, test_standard_acc, test_attack_loss,
                    test_attack_acc, dir):
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
    torch.save(state, dir)


def tb_writer_adjust_alpha(writer, epoch, lr, train_fgsm_loss, train_fgsm_acc,
                  train_pgd_loss, train_pgd_acc, train_all_norm,
                  test_clean_loss, test_clean_acc, test_fgsm_loss, test_fgsm_acc,
                  test_pgd_loss, test_pgd_acc, test_all_norm):
    writer.add_scalars('loss',
                       {'train_fgsm': train_fgsm_loss, 'train_pgd': train_pgd_loss,
                        'test_clean': test_clean_loss, 'test_fgsm': test_fgsm_loss, 'test_pgd': test_pgd_loss},
                       epoch + 1)
    writer.add_scalars('accuracy',
                       {'train_fgsm': train_fgsm_acc, 'train_pgd': train_pgd_acc,
                        'test_clean': test_clean_acc, 'test_fgsm': test_fgsm_acc, 'test_pgd': test_pgd_acc},
                       epoch + 1)
    train_inputs_grad_norm_mean = train_all_norm.mean()
    train_inputs_grad_norm_median = np.median(train_all_norm)
    train_inputs_grad_norm_std = train_all_norm.std()
    test_inputs_grad_norm_mean = test_all_norm.mean()
    test_inputs_grad_norm_median = np.median(test_all_norm)
    test_inputs_grad_norm_std = test_all_norm.std()
    writer.add_scalars('inputs_grad_norm_mean', {'train': train_inputs_grad_norm_mean, 'test': test_inputs_grad_norm_mean}, epoch + 1)
    writer.add_scalars('inputs_grad_norm_median', {'train': train_inputs_grad_norm_median, 'test': test_inputs_grad_norm_median}, epoch + 1)
    writer.add_scalars('inputs_grad_norm_std', {'train': train_inputs_grad_norm_std, 'test': test_inputs_grad_norm_std}, epoch + 1)
    writer.add_histogram('inputs_grad_norm_train', train_all_norm, epoch + 1, bins='auto')
    writer.add_histogram('inputs_grad_norm_test', test_all_norm, epoch + 1, bins='auto')
    writer.add_scalar('learning rate', lr, epoch + 1)


def tb_writer(writer, epoch, lr, train_fgsm_loss, train_fgsm_acc,
                  train_pgd_loss, train_pgd_acc, train_deepfool_loss, train_deepfool_acc,
                  train_all_norm, train_df_loop, train_df_perturbation_norm,
                  test_clean_loss, test_clean_acc, test_fgsm_loss, test_fgsm_acc,
                  test_pgd_loss, test_pgd_acc, test_deepfool_loss, test_deepfool_acc,
                  test_all_norm, test_df_loop, test_df_perturbation_norm):
    writer.add_scalars('loss',
                       {'train_fgsm': train_fgsm_loss, 'train_pgd': train_pgd_loss, 'train_deepfool': train_deepfool_loss,
                        'test_clean': test_clean_loss, 'test_fgsm': test_fgsm_loss, 'test_pgd': test_pgd_loss, 'test_deepfool': test_deepfool_loss},
                       epoch + 1)
    writer.add_scalars('accuracy',
                       {'train_fgsm': train_fgsm_acc, 'train_pgd': train_pgd_acc, 'train_deepfool': train_deepfool_acc,
                        'test_clean': test_clean_acc, 'test_fgsm': test_fgsm_acc, 'test_pgd': test_pgd_acc, 'test_deepfool': test_deepfool_acc},
                       epoch + 1)
    train_inputs_grad_norm_mean = train_all_norm.mean()
    train_inputs_grad_norm_median = np.median(train_all_norm)
    train_inputs_grad_norm_std = train_all_norm.std()
    test_inputs_grad_norm_mean = test_all_norm.mean()
    test_inputs_grad_norm_median = np.median(test_all_norm)
    test_inputs_grad_norm_std = test_all_norm.std()
    writer.add_scalars('inputs_grad_norm_mean', {'train': train_inputs_grad_norm_mean, 'test': test_inputs_grad_norm_mean}, epoch + 1)
    writer.add_scalars('inputs_grad_norm_median', {'train': train_inputs_grad_norm_median, 'test': test_inputs_grad_norm_median}, epoch + 1)
    writer.add_scalars('inputs_grad_norm_std', {'train': train_inputs_grad_norm_std, 'test': test_inputs_grad_norm_std}, epoch + 1)
    writer.add_histogram('inputs_grad_norm_train', train_all_norm, epoch + 1, bins='auto')
    writer.add_histogram('inputs_grad_norm_test', test_all_norm, epoch + 1, bins='auto')
    writer.add_scalar('learning rate', lr, epoch + 1)

    train_df_loop_mean = -1
    train_df_loop_median = -1
    train_df_loop_std = -1
    train_df_perturbation_mean = -1
    train_df_perturbation_median = -1
    train_df_perturbation_std = -1
    if train_df_loop.size > 1:
        train_df_loop_mean = train_df_loop.mean()
        train_df_loop_median = np.median(train_df_loop)
        train_df_loop_std = train_df_loop.std()
        train_df_perturbation_mean = train_df_perturbation_norm.mean()
        train_df_perturbation_median = np.median(train_df_perturbation_norm)
        train_df_perturbation_std = train_df_perturbation_norm.std()

    test_df_loop_mean = -1
    test_df_loop_median = -1
    test_df_loop_std = -1
    test_df_perturbation_mean = -1
    test_df_perturbation_median = -1
    test_df_perturbation_std = -1
    if test_df_loop.size > 1:
        test_df_loop_mean = test_df_loop.mean()
        test_df_loop_median = np.median(test_df_loop)
        test_df_loop_std = test_df_loop.std()
        test_df_perturbation_mean = test_df_perturbation_norm.mean()
        test_df_perturbation_median = np.median(test_df_perturbation_norm)
        test_df_perturbation_std = test_df_perturbation_norm.std()
    writer.add_scalars('df_loop_mean', {'train': train_df_loop_mean, 'test': test_df_loop_mean}, epoch + 1)
    writer.add_scalars('df_loop_median', {'train': train_df_loop_median, 'test': test_df_loop_median}, epoch + 1)
    writer.add_scalars('df_loop_std', {'train': train_df_loop_std, 'test': test_df_loop_std}, epoch + 1)
    writer.add_scalars('df_perturbation_mean', {'train': train_df_perturbation_mean, 'test': test_df_perturbation_mean}, epoch + 1)
    writer.add_scalars('df_perturbation_median', {'train': train_df_perturbation_median, 'test': test_df_perturbation_median}, epoch + 1)
    writer.add_scalars('df_perturbation_std', {'train': train_df_perturbation_std, 'test': test_df_perturbation_std}, epoch + 1)
    writer.add_histogram('train_df_loop', train_df_loop, epoch + 1, bins='auto')
    writer.add_histogram('test_df_loop', test_df_loop, epoch + 1, bins='auto')
    writer.add_histogram('train_df_perturbation_norm', train_df_perturbation_norm, epoch + 1, bins='auto')
    writer.add_histogram('test_df_perturbation_norm', test_df_perturbation_norm, epoch + 1, bins='auto')


def log_resumed_info(checkpoint, logger):
    resumed_epoch = checkpoint['epoch']
    resumed_train_loss = checkpoint['train_loss']
    resumed_train_acc = checkpoint['train_acc']
    resumed_test_standard_loss = checkpoint['test_standard_loss']
    resumed_test_standard_acc = checkpoint['test_standard_acc']
    resumed_test_attack_loss = checkpoint['test_attack_loss']
    resumed_test_attack_acc = checkpoint['test_attack_acc']

    logger.info(
        f"finetune from epoch {resumed_epoch}, train loss {resumed_train_loss}, train acc {resumed_train_acc}, Test "
        f"Standard Loss {resumed_test_standard_loss}, Test Standard Acc {resumed_test_standard_acc}, Test Attack Loss "
        f"{resumed_test_attack_loss}, Test Attack Acc {resumed_test_attack_acc}")
