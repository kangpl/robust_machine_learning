import torch
import torch.nn.functional as F


def cure(model, inputs, target, h=1.5, device='cuda'): #increase first 5 slowly,  adam,  always from finetuning,  60 epoches
    # get z
    model.eval()
    delta = torch.zeros_like(inputs, requires_grad=True).to(device)
    output = model(inputs + delta)
    loss = F.cross_entropy(output, target, reduction='sum') #take the sum
    grad = torch.autograd.grad(loss, delta, create_graph=True)[0]

    grad_sign = torch.sign(grad.detach())
    z = (grad_sign + 1e-7) / (grad_sign.view(grad_sign.shape[0], -1).norm(dim=1)[:, None, None, None] + 1e-7)
    hz = (h * z).requires_grad_()

    output_hz = model(inputs + hz)
    loss_hz = F.cross_entropy(output_hz, target, reduction='sum')
    grad_hz = torch.autograd.grad(loss_hz, hz, create_graph=True)[0]

    grad_diff = grad_hz - grad
    # lr = torch.pow(grad_diff.view(grad_diff.shape[0], -1).norm(dim=1), 2)
    lr = grad_diff.view(grad_diff.shape[0], -1).norm(dim=1)

    model.train()
    return lr.mean()

