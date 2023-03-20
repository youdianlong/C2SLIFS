import numpy as np
import torch

def convert_logits_to_sigmoid(w, tau=1.0):

    d = w.shape[0]
    sigmoid_w = torch.sigmoid(w / tau)
    sigmoid_w = sigmoid_w * (1. - torch.eye(d, device=sigmoid_w.device))    # Mask diagonal to 0

    return sigmoid_w

def compute_acyclicity(w):
    """
    Parameters
    ----------
    w: torch.Tensor
    """

    return torch.trace(torch.matrix_exp(w * w)) - w.shape[0]

def sample_gumbel(shape, eps=1e-20, seed=0, device=None):

    eps = torch.tensor(eps, device=device)
    torch.manual_seed(seed)
    u = torch.rand(shape, device=device)
    u = -torch.log(-torch.log(u + eps) + eps)
    u[np.arange(shape[0]), np.arange(shape[0])] = 0

    return u

def gumbel_sigmoid(logits, temperature, seed, device):

    gumbel_softmax_sample = (logits
                 + sample_gumbel(logits.shape, seed=seed, device=device)
                 - sample_gumbel(logits.shape, seed=seed + 1, device=device))
    y = torch.sigmoid(gumbel_softmax_sample / temperature)

    return y

def generate_upper_triangle_indices(d, device=None):

    if d == 1:
        return torch.zeros(1, dtype=torch.int64, device=device)
    mat = np.arange((d)**2).reshape(d, d)
    mat = torch.tensor(mat, dtype=torch.long, device=device)
    target_indices = mat[torch.triu(mat, 1) != 0]

    return target_indices

def callback_after_training(w_logits, temperature, graph_thresh):

    d = w_logits.shape[0]
    w_final_weight = convert_logits_to_sigmoid(w_logits, temperature)
    w_final = w_final_weight.clone()  # final graph
    w_final[w_final <= graph_thresh] = 0    # Thresholding
    w_final[w_final > graph_thresh] = 1
    w_final[np.arange(d), np.arange(d)] = 0    # Mask diagonal to 0

    return w_final, w_final_weight

def tensor_description(var):
    """
    Returns a compact and informative string about a tensor.

    Parameters
    ----------
    var: A tensor variable.

    Returns

    """
    description = '(' + str(var.dtype.name) + ' '
    sizes = var.get_shape()
    for i, size in enumerate(sizes):
        description += str(size)
        if i < len(sizes) - 1:
            description += 'x'
    description += ')'

    return description