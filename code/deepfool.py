import copy

from utils.util import *


def deepfool(model, inputs, target, epsilon, num_classes=2, overshoot=0.02, max_iter=50, device='cuda'):
    """
       :param epsilon:
       :param inputs:
       :param target:
       :param device:
       :param model: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    w = torch.zeros(inputs.shape).to(device)
    r_tot = torch.zeros(inputs.shape).to(device)

    loop_i = 0
    pert_inputs = copy.deepcopy(inputs).requires_grad_()
    while loop_i < max_iter:
        output = model(pert_inputs)
        index = torch.where(output.max(1)[1] == target)[0]
        if loop_i == 0:
            I = torch.flip(output.argsort(), (-1,))[:, :num_classes]
        if len(index) == 0:
            break

        pert = torch.full((target.shape[0],), float("Inf")).to(device)

        ori_t = output.gather(1, target.view(-1, 1))
        ori_grad = torch.autograd.grad(ori_t.sum(), pert_inputs, create_graph=True)[0].detach()

        for k in range(1, num_classes):
            cur_t = output.gather(1, I[:, k].view(-1, 1))
            cur_grad = torch.autograd.grad(cur_t.sum(), pert_inputs, create_graph=True)[0].detach()

            # set new w_k and new f_k
            w_k = cur_grad - ori_grad
            f_k = (cur_t - ori_t).detach().flatten()

            pert_k = abs(f_k) / torch.norm(w_k.view(w_k.shape[0], -1), dim=-1)

            # determine which w_k to use
            valid_index = torch.where(pert_k < pert)[0]
            inter_index = np.intersect1d(valid_index.cpu(), index.cpu())
            pert[inter_index] = pert_k[inter_index]
            w[inter_index] = w_k[inter_index]

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert[index, None, None, None] + 1e-4) * w[index] / torch.norm(w[index].view(w[index].shape[0], -1),
                                                                             dim=1).view(-1, 1, 1, 1)
        r_tot[index] = r_tot[index] + r_i
        r_tot = clamp(r_tot, -epsilon, epsilon)
        pert_inputs = clamp(inputs + (1 + overshoot) * r_tot, lower_limit, upper_limit).requires_grad_()
        loop_i += 1

    return pert_inputs.detach()
