import torch
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients


def cure(model, inputs, target, h=1.5, device='cuda'):
    # get z
    model.train()
    inputs.requires_grad_()
    output = model(inputs)
    loss = F.cross_entropy(output, target, reduction='sum')
    grad = torch.autograd.grad(loss, inputs)[0]
    grad_sign = torch.sign(grad.detach())
    hz = h * (grad_sign + 1e-7) / (grad_sign.view(grad_sign.shape[0], -1).norm(dim=1)[:, None, None, None] + 1e-7)

    model.train()
    output = model(inputs)
    output_hz = model(inputs + hz)
    loss = F.cross_entropy(output, target, reduction='none')
    loss_hz = F.cross_entropy(output_hz, target, reduction='none')
    grad_diff = torch.autograd.grad(loss_hz - loss, inputs, grad_outputs=torch.ones(target.size()).to(device), create_graph=True)[0]

    reg = grad_diff.view(grad_diff.shape[0], -1).norm(dim=1)
    model.zero_grad()
    inputs.requires_grad = False

    return reg.mean()

