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
# cifar10_std = (0.2023, 0.1994, 0.2010)

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


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, device, norm="l_inf", random_start=True,
               early_stop=False):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if random_start:
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
        delta_init = torch.zeros_like(X, requires_grad=True).to(device)

    output = model(X + delta_init)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, delta_init, create_graph=True if backprop else False)[0]
    if not backprop:
        grad = grad.detach()
    return grad


def get_input_grad_v2(model, X, y):
    X.requires_grad_()
    output = model(X)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, X, create_graph=False)[0]
    X.requires_grad = False
    return grad.detach(), output, loss


def cal_cos_similarity(grad1, grad2, grad1_norm, grad2_norm):
    grads_nnz_idx = (grad1_norm != 0) * (grad2_norm != 0)
    grad1, grad2 = grad1[grads_nnz_idx], grad2[grads_nnz_idx]
    grad1_norms, grad2_norms = grad1_norm[grads_nnz_idx], grad2_norm[grads_nnz_idx]
    grad1_normalized = grad1 / grad1_norms[:, None, None, None]
    grad2_normalized = grad2 / grad2_norms[:, None, None, None]
    cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
    return cos


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


def tb_writer_cure(writer, epoch, lr, train_loss, train_acc, train_reg, test_clean_loss, test_clean_acc, test_pgd_loss,
                   test_pgd_acc):
    writer.add_scalars('loss', {'train': train_loss, 'test_clean': test_clean_loss, 'test_pgd': test_pgd_loss},
                       epoch + 1)
    writer.add_scalars('accuracy', {'train': train_acc, 'test_clean': test_clean_acc, 'test_pgd': test_pgd_acc},
                       epoch + 1)
    writer.add_scalar('learning rate', lr, epoch + 1)
    writer.add_scalar('curvature regularization', train_reg, epoch + 1)


def tb_writer_clean(writer, epoch, lr, train_loss, train_acc, train_deepfool_loss, train_deepfool_acc,
                    train_input_grad_norm, train_df_loop, train_df_perturbation_norm, train_df_grad_norm, train_cos,
                    test_clean_loss, test_clean_acc, test_deepfool_loss, test_deepfool_acc,
                    test_input_grad_norm, test_df_loop, test_df_perturbation_norm, test_df_grad_norm, test_cos):
    writer.add_scalars('loss',
                       {'train_clean': train_loss,
                        'train_deepfool': train_deepfool_loss,
                        'test_clean': test_clean_loss,
                        'test_deepfool': test_deepfool_loss},
                       epoch + 1)
    writer.add_scalars('accuracy',
                       {'train_clean': train_acc,
                        'train_deepfool': train_deepfool_acc,
                        'test_clean': test_clean_acc,
                        'test_deepfool': test_deepfool_acc},
                       epoch + 1)
    train_inputs_grad_norm_mean, train_inputs_grad_norm_median, train_inputs_grad_norm_std = train_input_grad_norm.mean(), np.median(
        train_input_grad_norm), train_input_grad_norm.std()
    train_df_grad_norm_mean, train_df_grad_norm_median, train_df_grad_norm_std = train_df_grad_norm.mean(), np.median(
        train_df_grad_norm), train_df_grad_norm.std()
    train_cos_mean, train_cos_median, train_cos_std = train_cos.mean(), np.median(train_cos), train_cos.std()
    test_inputs_grad_norm_mean, test_inputs_grad_norm_median, test_inputs_grad_norm_std = test_input_grad_norm.mean(), np.median(
        test_input_grad_norm), test_input_grad_norm.std()
    test_df_grad_norm_mean, test_df_grad_norm_median, test_df_grad_norm_std = test_df_grad_norm.mean(), np.median(
        test_df_grad_norm), test_df_grad_norm.std()
    test_cos_mean, test_cos_median, test_cos_std = test_cos.mean(), np.median(test_cos), test_cos.std()

    writer.add_scalars('grad_norm_mean',
                       {'train_clean': train_inputs_grad_norm_mean, 'train_df': train_df_grad_norm_mean,
                        'test_clean': test_inputs_grad_norm_mean, 'test_df': test_df_grad_norm_mean}, epoch + 1)
    writer.add_scalars('grad_norm_median',
                       {'train_clean': train_inputs_grad_norm_median, 'train_df': train_df_grad_norm_median,
                        'test_clean_inputs': test_inputs_grad_norm_median, 'test_df': test_df_grad_norm_median},
                       epoch + 1)
    writer.add_scalars('grad_norm_std',
                       {'train_clean': train_inputs_grad_norm_std, 'train_df': train_df_grad_norm_std,
                        'test_clean_inputs': test_inputs_grad_norm_std,
                        'test_df': test_df_grad_norm_std},
                       epoch + 1)
    writer.add_scalars('cos_similarity_mean', {'train': train_cos_mean, 'test': test_cos_mean}, epoch + 1)
    writer.add_scalars('cos_similarity_median', {'train': train_cos_median, 'test': test_cos_median}, epoch + 1)
    writer.add_scalars('cos_similarity_std', {'train': train_cos_std, 'test': test_cos_std}, epoch + 1)
    writer.add_scalar('learning rate', lr, epoch + 1)

    train_df_loop_mean, train_df_loop_median, train_df_loop_std = train_df_loop.mean(), np.median(
        train_df_loop), train_df_loop.std()
    train_df_perturbation_mean, train_df_perturbation_median, train_df_perturbation_std = train_df_perturbation_norm.mean(), np.median(
        train_df_perturbation_norm), train_df_perturbation_norm.std()

    test_df_loop_mean, test_df_loop_median, test_df_loop_std = test_df_loop.mean(), np.median(
        test_df_loop), test_df_loop.std()
    test_df_perturbation_mean, test_df_perturbation_median, test_df_perturbation_std = test_df_perturbation_norm.mean(), np.median(
        test_df_perturbation_norm), test_df_perturbation_norm.std()
    writer.add_scalars('df_loop_mean', {'train': train_df_loop_mean, 'test': test_df_loop_mean}, epoch + 1)
    writer.add_scalars('df_loop_median', {'train': train_df_loop_median, 'test': test_df_loop_median}, epoch + 1)
    writer.add_scalars('df_loop_std', {'train': train_df_loop_std, 'test': test_df_loop_std}, epoch + 1)
    writer.add_scalars('df_perturbation_mean',
                       {'train': train_df_perturbation_mean, 'test': test_df_perturbation_mean},
                       epoch + 1)
    writer.add_scalars('df_perturbation_median',
                       {'train': train_df_perturbation_median, 'test': test_df_perturbation_median}, epoch + 1)
    writer.add_scalars('df_perturbation_std',
                       {'train': train_df_perturbation_std, 'test': test_df_perturbation_std},
                       epoch + 1)


def tb_writer_(writer, epoch, lr, train_attack_loss, train_attack_acc, metrics):
    writer.add_scalars('loss',
                       {'train_attack': train_attack_loss,
                        'train_clean_eval' : metrics['train_clean_loss'] / metrics['train_total'],
                        'train_fgsm_eval': metrics['train_fgsm_loss'] / metrics['train_total'],
                        'train_pgd': metrics['train_pgd_loss'] / metrics['train_total'],
                        'test_clean': metrics['test_clean_loss'] / metrics['test_total'],
                        'test_fgsm':  metrics['test_fgsm_loss'] / metrics['test_total'],
                        'test_pgd':  metrics['test_pgd_loss'] / metrics['test_total']}, epoch)
    writer.add_scalars('accuracy',
                       {'train_attack': train_attack_acc,
                        'train_clean_eval': 100. * metrics['train_clean_correct'] / metrics['train_total'],
                        'train_fgsm_eval': 100. * metrics['train_fgsm_correct'] / metrics['train_total'],
                        'train_pgd': 100. * metrics['train_pgd_correct'] / metrics['train_total'],
                        'test_clean': 100. * metrics['test_clean_correct'] / metrics['test_total'],
                        'test_fgsm': 100. * metrics['test_fgsm_correct'] / metrics['test_total'],
                        'test_pgd': 100. * metrics['test_pgd_correct'] / metrics['test_total']}, epoch)

    train_df_loop = np.concatenate(metrics['train_df_loop'])
    train_df_perturbation_norm = np.concatenate(metrics['train_df_perturbation_norm'])
    train_df_grad_norm = np.concatenate(metrics['train_df_grad_norm'])
    train_clean_grad_norm = np.concatenate(metrics['train_clean_grad_norm'])
    train_fgsm_grad_norm = np.concatenate(metrics['train_fgsm_grad_norm'])
    train_pgd_grad_norm = np.concatenate(metrics['train_pgd_grad_norm'])
    train_cos_clean_df = np.concatenate(metrics['train_cos_clean_df'])
    train_cos_clean_fgsm = np.concatenate(metrics['train_cos_clean_fgsm'])
    train_cos_clean_pgd = np.concatenate(metrics['train_cos_clean_pgd'])
    train_cos_fgsm_pgd = np.concatenate(metrics['train_cos_fgsm_pgd'])

    test_df_loop = np.concatenate(metrics['test_df_loop'])
    test_df_perturbation_norm = np.concatenate(metrics['test_df_perturbation_norm'])
    test_df_grad_norm = np.concatenate(metrics['test_df_grad_norm'])
    test_clean_grad_norm = np.concatenate(metrics['test_clean_grad_norm'])
    test_fgsm_grad_norm = np.concatenate(metrics['test_fgsm_grad_norm'])
    test_pgd_grad_norm = np.concatenate(metrics['test_pgd_grad_norm'])
    test_cos_clean_df = np.concatenate(metrics['test_cos_clean_df'])
    test_cos_clean_fgsm = np.concatenate(metrics['test_cos_clean_fgsm'])
    test_cos_clean_pgd = np.concatenate(metrics['test_cos_clean_pgd'])
    test_cos_fgsm_pgd = np.concatenate(metrics['test_cos_fgsm_pgd'])

    train_inputs_grad_norm_mean, train_inputs_grad_norm_median, train_inputs_grad_norm_std = train_clean_grad_norm.mean(), np.median(
        train_clean_grad_norm), train_clean_grad_norm.std()
    train_df_grad_norm_mean, train_df_grad_norm_median, train_df_grad_norm_std = train_df_grad_norm.mean(), np.median(
        train_df_grad_norm), train_df_grad_norm.std()
    train_fgsm_grad_norm_mean, train_fgsm_grad_norm_median, train_fgsm_grad_norm_std = train_fgsm_grad_norm.mean(), np.median(
        train_fgsm_grad_norm), train_fgsm_grad_norm.std()
    train_pgd_grad_norm_mean, train_pgd_grad_norm_median, train_pgd_grad_norm_std = train_pgd_grad_norm.mean(), np.median(
        train_pgd_grad_norm), train_pgd_grad_norm.std()
    train_cos_clean_df_mean, train_cos_clean_df_median, train_cos_clean_df_std = train_cos_clean_df.mean(), np.median(
        train_cos_clean_df), train_cos_clean_df.std()
    train_cos_clean_fgsm_mean, train_cos_clean_fgsm_median, train_cos_clean_fgsm_std = train_cos_clean_fgsm.mean(), np.median(
        train_cos_clean_fgsm), train_cos_clean_fgsm.std()
    train_cos_clean_pgd_mean, train_cos_clean_pgd_median, train_cos_clean_pgd_std = train_cos_clean_pgd.mean(), np.median(
        train_cos_clean_pgd), train_cos_clean_pgd.std()
    train_cos_fgsm_pgd_mean, train_cos_fgsm_pgd_median, train_cos_fgsm_pgd_std = train_cos_fgsm_pgd.mean(), np.median(
        train_cos_fgsm_pgd), train_cos_fgsm_pgd.std()

    test_inputs_grad_norm_mean, test_inputs_grad_norm_median, test_inputs_grad_norm_std = test_input_grad_norm.mean(), np.median(
        test_clean_grad_norm), test_clean_grad_norm.std()
    test_df_grad_norm_mean, test_df_grad_norm_median, test_df_grad_norm_std = test_df_grad_norm.mean(), np.median(
        test_df_grad_norm), test_df_grad_norm.std()
    test_fgsm_grad_norm_mean, test_fgsm_grad_norm_median, test_fgsm_grad_norm_std = test_fgsm_grad_norm.mean(), np.median(
        test_fgsm_grad_norm), test_fgsm_grad_norm.std()
    test_pgd_grad_norm_mean, test_pgd_grad_norm_median, test_pgd_grad_norm_std = test_pgd_grad_norm.mean(), np.median(
        test_pgd_grad_norm), test_pgd_grad_norm.std()
    test_cos_clean_df_mean, test_cos_clean_df_median, test_cos_clean_df_std = test_cos_clean_df.mean(), np.median(
        test_cos_clean_df), test_cos_clean_df.std()
    test_cos_clean_fgsm_mean, test_cos_clean_fgsm_median, test_cos_clean_fgsm_std = test_cos_clean_fgsm.mean(), np.median(
        test_cos_clean_fgsm), test_cos_clean_fgsm.std()
    test_cos_clean_pgd_mean, test_cos_clean_pgd_median, test_cos_clean_pgd_std = test_cos_clean_pgd.mean(), np.median(
        test_cos_clean_pgd), test_cos_clean_pgd.std()
    test_cos_fgsm_pgd_mean, test_cos_fgsm_pgd_median, test_cos_fgsm_pgd_std = test_cos_fgsm_pgd.mean(), np.median(
        test_cos_fgsm_pgd), test_cos_fgsm_pgd.std()

    writer.add_scalars('grad_norm_mean',
                       {'train_clean': train_inputs_grad_norm_mean, 'train_df': train_df_grad_norm_mean,
                        'train_fgsm': train_fgsm_grad_norm_mean, 'train_pgd': train_pgd_grad_norm_mean,
                        'test_clean': test_inputs_grad_norm_mean, 'test_df': test_df_grad_norm_mean,
                        'test_fgsm': test_fgsm_grad_norm_mean, 'test_pgd': test_pgd_grad_norm_mean}, epoch)
    writer.add_scalars('grad_norm_median',
                       {'train_clean': train_inputs_grad_norm_median, 'train_df': train_df_grad_norm_median,
                        'train_fgsm': train_fgsm_grad_norm_median, 'train_pgd': train_pgd_grad_norm_median,
                        'test_clean_inputs': test_inputs_grad_norm_median, 'test_df': test_df_grad_norm_median,
                        'test_fgsm': test_fgsm_grad_norm_median, 'test_pgd': test_pgd_grad_norm_median},
                       epoch)
    writer.add_scalars('grad_norm_std',
                       {'train_clean': train_inputs_grad_norm_std, 'train_df': train_df_grad_norm_std,
                        'train_fgsm': train_fgsm_grad_norm_std, 'train_pgd': train_pgd_grad_norm_std,
                        'test_clean_inputs': test_inputs_grad_norm_std, 'test_df': test_df_grad_norm_std,
                        'test_fgsm': test_fgsm_grad_norm_std, 'test_pgd': test_pgd_grad_norm_std},
                       epoch)
    writer.add_scalars('cos_similarity_mean',
                       {'train_clean_df': train_cos_clean_df_mean, 'train_clean_fgsm': train_cos_clean_fgsm_mean,
                        'train_clean_pgd': train_cos_clean_pgd_mean, 'train_fgsm_pgd': train_cos_fgsm_pgd_mean,
                        'test_clean_df': test_cos_clean_df_mean, 'test_clean_fgsm': test_cos_clean_fgsm_mean,
                        'test_clean_pgd': test_cos_clean_pgd_mean, 'test_fgsm_pgd': test_cos_fgsm_pgd_mean},
                       epoch)
    writer.add_scalars('cos_similarity_median',
                       {'train_clean_df': train_cos_clean_df_median, 'train_clean_fgsm': train_cos_clean_fgsm_median,
                        'train_clean_pgd': train_cos_clean_pgd_median, 'train_fgsm_pgd': train_cos_fgsm_pgd_median,
                        'test_clean_df': test_cos_clean_df_median, 'test_clean_fgsm': test_cos_clean_fgsm_median,
                        'test_clean_pgd': test_cos_clean_pgd_median, 'test_fgsm_pgd': test_cos_fgsm_pgd_median},
                       epoch)
    writer.add_scalars('cos_similarity_std',
                       {'train_clean_df': train_cos_clean_df_std, 'train_clean_fgsm': train_cos_clean_fgsm_std,
                        'train_clean_pgd': train_cos_clean_pgd_std, 'train_fgsm_pgd': train_cos_fgsm_pgd_std,
                        'test_clean_df': test_cos_clean_df_std, 'test_clean_fgsm': test_cos_clean_fgsm_std,
                        'test_clean_pgd': test_cos_clean_pgd_std, 'test_fgsm_pgd': test_cos_fgsm_pgd_std},
                       epoch)
    writer.add_scalar('learning rate', lr, epoch)

    train_df_loop_mean, train_df_loop_median, train_df_loop_std = train_df_loop.mean(), np.median(
        train_df_loop), train_df_loop.std()
    train_df_perturbation_mean, train_df_perturbation_median, train_df_perturbation_std = train_df_perturbation_norm.mean(), np.median(
        train_df_perturbation_norm), train_df_perturbation_norm.std()

    test_df_loop_mean, test_df_loop_median, test_df_loop_std = test_df_loop.mean(), np.median(
        test_df_loop), test_df_loop.std()
    test_df_perturbation_mean, test_df_perturbation_median, test_df_perturbation_std = test_df_perturbation_norm.mean(), np.median(
        test_df_perturbation_norm), test_df_perturbation_norm.std()
    writer.add_scalars('df_loop_mean', {'train': train_df_loop_mean, 'test': test_df_loop_mean}, epoch)
    writer.add_scalars('df_loop_median', {'train': train_df_loop_median, 'test': test_df_loop_median}, epoch)
    writer.add_scalars('df_loop_std', {'train': train_df_loop_std, 'test': test_df_loop_std}, epoch)
    writer.add_scalars('df_perturbation_mean', {'train': train_df_perturbation_mean, 'test': test_df_perturbation_mean},
                       epoch)
    writer.add_scalars('df_perturbation_median',
                       {'train': train_df_perturbation_median, 'test': test_df_perturbation_median}, epoch)
    writer.add_scalars('df_perturbation_std', {'train': train_df_perturbation_std, 'test': test_df_perturbation_std},
                       epoch)


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


# def evaluation(args, model, inputs, targets, metrics, dataset='train'):
#     # clean
#     clean_grad, clean_outputs, clean_loss = get_input_grad_v2(model, inputs, targets)
#     clean_grad_norm = clean_grad.view(clean_grad.shape[0], -1).norm(dim=1)
#
#     # fgsm
#     fgsm_delta = attack_pgd(model, inputs, targets, args.test_epsilon, args.eval_fgsm_alpha, 1, 1, args.device).detach()
#     fgsm_grad, fgsm_outputs, fgsm_loss = get_input_grad_v2(model,
#                                                            clamp(inputs + fgsm_delta, lower_limit, upper_limit),
#                                                            targets)
#     fgsm_grad_norm = fgsm_grad.view(fgsm_grad.shape[0], -1).norm(dim=1)
#
#     # pgd
#     pgd_delta = attack_pgd(model, inputs, targets, args.eval_epsilon, args.eval_pgd_alpha,
#                            args.eval_pgd_attack_iters, args.eval_pgd_restarts, args.device,
#                            early_stop=True).detach()
#     pgd_grad, pgd_outputs, pgd_loss = get_input_grad_v2(model, clamp(inputs + pgd_delta, lower_limit, upper_limit),
#                                                         targets)
#     pgd_grad_norm = pgd_grad.view(pgd_grad.shape[0], -1).norm(dim=1)
#
#     # deepfool attack
#     pert_inputs, loop, perturbation = deepfool(model, inputs, num_classes=args.eval_deepfool_classes_num,
#                                                max_iter=args.eval_deepfool_max_iter, device=args.device)
#     deepfool_grad, deepfool_outputs, deepfool_loss = get_input_grad_v2(model, pert_inputs, targets)
#     deepfool_grad_norm = deepfool_grad.view(deepfool_grad.shape[0], -1).norm(dim=1)
#
#     # calculate cosine
#     cos_clean_df = cal_cos_similarity(clean_grad, deepfool_grad, clean_grad_norm, deepfool_grad_norm)
#     cos_clean_fgsm = cal_cos_similarity(clean_grad, fgsm_grad, clean_grad_norm, fgsm_grad_norm)
#     cos_clean_pgd = cal_cos_similarity(clean_grad, pgd_grad, clean_grad_norm, pgd_grad_norm)
#     cos_fgsm_pgd = cal_cos_similarity(fgsm_grad, pgd_grad, fgsm_grad_norm, pgd_grad_norm)
#
#
#     metrics[dataset + '_clean_loss'] += clean_loss.item() * targets.size(0)
#     metrics[dataset + '_clean_correct'] += (clean_outputs.max(1)[1] == targets).sum().item()
#     metrics[dataset + '_clean_grad_norm'].append(clean_grad_norm.cpu().numpy())
#
#     metrics[dataset + '_fgsm_loss'] += fgsm_loss.item() * targets.size(0)
#     metrics[dataset + '_fgsm_correct'] += (fgsm_outputs.max(1)[1] == targets).sum().item()
#     metrics[dataset + '_fgsm_grad_norm'].append(fgsm_grad_norm.cpu().numpy())
#
#     metrics[dataset + '_pgd_loss'] += pgd_loss.item() * targets.size(0)
#     metrics[dataset + '_pgd_correct'] += (pgd_outputs.max(dim=1)[1] == targets).sum().item()
#     metrics[dataset + '_pgd_grad_norm'].append(pgd_grad_norm.cpu().numpy())
#
#     metrics[dataset + '_df_loop'].append(loop.cpu().numpy())
#     metrics[dataset + '_df_perturbation_norm'].append(perturbation.cpu().numpy())
#     metrics[dataset + '_df_grad_norm'].append(deepfool_grad_norm.cpu().numpy())
#
#     metrics[dataset + '_cos_clean_df'].append(cos_clean_df.cpu().numpy())
#     metrics[dataset + '_cos_clean_fgsm'].append(cos_clean_fgsm.cpu().numpy())
#     metrics[dataset + '_cos_clean_pgd'].append(cos_clean_pgd.cpu().numpy())
#     metrics[dataset + '_cos_fgsm_pgd'].append(cos_fgsm_pgd.cpu().numpy())
#
#     metrics[dataset + '_total'] += targets.size(0)
