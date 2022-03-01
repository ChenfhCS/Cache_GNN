import numpy as np
import torch
from dgl import DGLGraph

def Approx_prefix(input_features, parameter=10):
    approx_results = input_features['features'][:, :parameter]
    return approx_results