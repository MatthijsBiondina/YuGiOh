import torch
from torch import Tensor

from src.utils.tools import pyout, contains_nan


def batchwise_triplet_loss(A, P, eps=1e-6, margin=2.0):
    N = A.size(0)

    d_pos = ((A - P) ** 2).sum(dim=1) ** .5
    d_neg = ((A - torch.roll(P, 1, dims=(0,))) ** 2).sum(dim=1) ** .5

    loss = torch.clamp(d_pos - d_neg + margin, min=0.)

    return torch.sum(loss)

    pyout()

    # dmatrix: Tensor = ((A[:, None, :] - P[None, :, :]) ** 2).mean(dim=2) ** .5
    # d_pos = dmatrix.diagonal()
    # d_neg = dmatrix[~ torch.eye(N, dtype=torch.bool)].view(N, N - 1)
    #
    # d: Tensor = d_pos[:, None] / (d_pos[:, None] + d_neg + eps)
    #
    # with torch.no_grad():
    #     contains_nan(d)
    #
    # return d.sum()


def semi_hard_triplet_loss(A, P, margin=0.2):
    dmatrix = ((A[:, None, :] - P[None, :, :]) ** 2).sum(dim=2) ** .5
    d_pos = dmatrix.diagonal()

    attraction_loss = torch.maximum(d_pos, torch.full_like(d_pos, margin))

    with torch.no_grad():
        too_close_mask = (dmatrix < d_pos[:, None] + margin)
        eye_mask = torch.eye(dmatrix.size(0), dtype=torch.bool).to(too_close_mask.device)
        too_close_mask = too_close_mask & ~ eye_mask
        vals = torch.zeros_like(dmatrix)
        vals[too_close_mask] = dmatrix[too_close_mask]

        neg_idx = torch.argmax(vals, dim=1)
        all_good = torch.all(~ too_close_mask, dim=1)

    d_neg = dmatrix[torch.arange(dmatrix.size(0)), neg_idx]

    repulsion_loss = d_pos + 0.2 - d_neg
    repulsion_loss[all_good] = 0.

    return torch.sum(attraction_loss + repulsion_loss)
