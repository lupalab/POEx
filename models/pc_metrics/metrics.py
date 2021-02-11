import torch
import numpy as np

from .StructuralLosses.match_cost import match_cost
from .StructuralLosses.nn_distance import nn_distance

def distChamfer(x, y):
    '''
    x, y: [B,N,d]
    '''
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dl, dr = nn_distance(x, y)
    d = dl.mean(dim=1) + dr.mean(dim=1)

    return d.data.cpu().numpy()

def distEMD(x, y):
    B, N, N_ref = x.shape[0], x.shape[1], y.shape[1]
    assert N == N_ref
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    emd = match_cost(x, y)
    emd_norm = emd / float(N)

    return emd_norm.data.cpu().numpy()
