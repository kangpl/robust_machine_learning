import copy

from utils.util import *


def deepfool(model, inputs, num_classes=2, overshoot=0.02, max_iter=50, norm_dist='l_2', device='cuda', random_start=False, norm_rs='l_2', epsilon=-1, early_stop=False, model_in_eval=False, alpha=1, df_clamp=False):
    """
       :param inputs:
       :param device:
       :param model: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    if model_in_eval:
        model.eval()
    w = torch.zeros(inputs.shape).to(device)
    r_tot = torch.zeros(inputs.shape).to(device)

    if random_start:
        if norm_rs == 'l_2':
            r_tot.normal_()
            n = r_tot.view(r_tot.size(0), -1).norm(dim=1)[:, None, None, None]
            r = torch.zeros_like(n).uniform_(0, 1)
            r_tot *= r / n * epsilon
        elif norm_rs == 'l_inf':
            for i in range(len(epsilon)):
                r_tot[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        else:
            raise ValueError
        r_tot = clamp(r_tot, lower_limit - inputs, upper_limit - inputs)

    output = model(inputs + r_tot)
    I = torch.flip(output.argsort(), (-1,))[:, :num_classes]
    labels = I[:, 0]

    pert_inputs = (copy.deepcopy(inputs) + r_tot).requires_grad_()
    loop_i = torch.zeros(inputs.shape[0]).to(device)
    while loop_i.max() < max_iter:
        output = model(pert_inputs)
        if early_stop:
            index = torch.where(output.max(1)[1] == labels)[0]
        else:
            index = slice(None, None, None)
        if not isinstance(index, slice) and len(index) == 0:
            break

        pert = torch.full((labels.shape[0],), float("Inf")).to(device)

        ori_t = output.gather(1, labels.view(-1, 1)).flatten()
        ori_grad = torch.autograd.grad(ori_t, pert_inputs, grad_outputs=torch.ones(labels.size()).to(device), create_graph=True)[0].detach()

        for k in range(1, num_classes):
            cur_t = output.gather(1, I[:, k].view(-1, 1)).flatten()
            cur_grad = torch.autograd.grad(cur_t, pert_inputs, grad_outputs=torch.ones(labels.size()).to(device), create_graph=True)[0].detach()

            # set new w_k and new f_k
            w_k = cur_grad - ori_grad
            f_k = (cur_t - ori_t).detach()

            if norm_dist == 'l_2':
                pert_k = abs(f_k) / torch.norm(w_k.view(w_k.shape[0], -1), p=2, dim=1)
            elif norm_dist == 'l_inf':
                pert_k = abs(f_k) / torch.norm(w_k.view(w_k.shape[0], -1), p=1, dim=1)
            else:
                raise ValueError

            # determine which w_k to use
            valid_index = torch.where(pert_k < pert)[0]
            if not isinstance(index, slice):
                valid_index = np.intersect1d(valid_index.cpu(), index.cpu())
            pert[valid_index] = pert_k[valid_index]
            w[valid_index] = w_k[valid_index]

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        if norm_dist == 'l_2':
            r_i = (pert[index, None, None, None] + 1e-4) * w[index] / torch.norm(w[index].view(w[index].shape[0], -1),
                                                                                 dim=1).view(-1, 1, 1, 1)
        elif norm_dist == 'l_inf':
            r_i = (pert[index, None, None, None] + 1e-9) * w[index].sign()
        else:
            raise ValueError

        r_tot[index] = r_tot[index] + r_i
        pert_inputs = (inputs + (1 + overshoot) * r_tot).requires_grad_()
        loop_i[index] += 1
    if df_clamp:
        r_tot = clamp(alpha * (1 + overshoot) * r_tot, -epsilon, epsilon)
    else:
        r_tot = alpha * (1 + overshoot) * r_tot
    pert_inputs = (inputs + r_tot)
    return pert_inputs.detach(), loop_i, r_tot

