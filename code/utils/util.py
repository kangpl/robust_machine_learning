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
cifar10_mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
cifar10_std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


upper_limit, lower_limit = 1, 0


class Normalize:
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __call__(self, data):
        return (data - self.mu) / self.std


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


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


def get_loader(args, logger, dataset='cifar10'):
    # Data
    logger.info('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        datasetf = datasets.CIFAR10
    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        datasetf = datasets.CIFAR100
    elif dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        datasetf = datasets.SVHN
    else:
        raise ValueError

    if dataset == 'cifar10' or dataset == 'cifar100':
        trainset = datasetf(root=args.dataset_path, train=True, download=True, transform=transform_train)
        testset = datasetf(root=args.dataset_path, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        trainset = datasetf(args.dataset_path, split='train', transform=transform_train, download=True)
        testset = datasetf(args.dataset_path, split='test', transform=transform_test, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def attack_pgd(model, X, y, normalize, epsilon, alpha, attack_iters, restarts, device, norm="l_inf", random_start=True,
               early_stop=False):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if random_start:
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
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
            output = model(normalize(X + delta))
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
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(normalize(X + delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


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

