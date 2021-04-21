import copy

from utils.util import *


def deepfool_train(model, inputs, normalize, overshoot=0.02, max_iter=50, norm_dist='l_inf', device='cuda',
                   random_start=True, epsilon=-1, early_stop=False):
    r_tot = torch.zeros(inputs.shape).to(device)
    delta = torch.zeros(inputs.shape).to(device)
    if random_start:
        delta.uniform_(-epsilon, epsilon)
        delta = clamp(delta, lower_limit - inputs, upper_limit - inputs)

    output = model(normalize(inputs + delta))
    I = torch.flip(output.argsort(), (-1,))[:, :2]
    labels = I[:, 0]

    pert_inputs = (copy.deepcopy(inputs) + delta).requires_grad_()
    loop_i = torch.zeros(inputs.shape[0]).to(device)
    while loop_i.max() < max_iter:
        output = model(normalize(pert_inputs))
        if early_stop:
            index = torch.where(output.max(1)[1] == labels)[0]
        else:
            index = slice(None, None, None)
        if not isinstance(index, slice) and len(index) == 0:
            break

        cur_t = output.gather(1, I[:, 1].view(-1, 1)).flatten()
        ori_t = output.gather(1, labels.view(-1, 1)).flatten()
        w = torch.autograd.grad(cur_t - ori_t, pert_inputs, grad_outputs=torch.ones(labels.size()).to(device),
                                create_graph=True)[0].detach()
        f = (cur_t - ori_t).detach()

        if norm_dist == 'l_2':
            pert = abs(f) / torch.norm(w.view(w.shape[0], -1), p=2, dim=1)
            r_i = (pert[index, None, None, None] + 1e-4) * w[index] / torch.norm(w[index].view(w[index].shape[0], -1),
                                                                                 dim=1).view(-1, 1, 1, 1)
        elif norm_dist == 'l_inf':
            pert = abs(f) / torch.norm(w.view(w.shape[0], -1), p=1, dim=1)
            r_i = (pert[index, None, None, None] + 1e-9) * w[index].sign()
        else:
            raise ValueError

        r_tot[index] = r_tot[index] + r_i
        pert_inputs = (inputs + delta + (1 + overshoot) * r_tot).requires_grad_()
        loop_i[index] += 1

    r_tot = delta + (1 + overshoot) * r_tot
    return loop_i, r_tot
