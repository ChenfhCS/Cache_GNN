import numpy as np
import torch
from dgl import DGLGraph

def Approx_prefix(input_features, parameter=0.5):
    scale = int(input_features['features'].size(1)*parameter)
    approx_results = input_features['features'][:, :scale]
    return approx_results