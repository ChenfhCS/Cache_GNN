import argparse, time, math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}


def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 cuda,
                 bias=True,
                 cache=False):
        super(GCNLayer, self).__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        # self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.reset_parameters()

        self.adj = g.adj().to_dense()
        self.activation = activation
        self.cache = cache

        # normalization
        degs = g.in_degrees().float()  # degrees of all nodes (1-dimension tensor)
        norm = torch.pow(degs, -0.5)  # 
        norm_tilde = torch.diag(norm) # 
        if cuda:
            norm_tilde.cuda()
        self.adj = torch.mm(torch.mm(norm_tilde, self.adj), norm_tilde)  # D^(-0/5) * A * D^(-0/5)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        # modified graph convolutional operations

        if self.cache == False:
            # step1: aggregation
            h = torch.mm(self.adj, h)
            cache_h = h

        # step2: dropout
        if self.dropout:
            h = self.dropout(h)

        # step3: reduce
        h = torch.mm(h, self.weight)

        # step4: activation
        if self.activation:
            h = self.activation(h)

        return h

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 features,
                 Cache,
                 cuda,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(g, in_feats, n_hidden, activation, dropout, cuda, cache=Cache))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(g, n_hidden, n_hidden, activation, dropout, cuda))
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, n_classes, None, dropout, cuda))

        self.Cache = Cache
        self.cuda = cuda
        if Cache:
            self.cache = self.cache_init(g, features, dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i == 0:
                if self.Cache:
                    h = layer(self.cache)
                else:
                    h = layer(h)
            else:
                h = layer(h)
        return h
    
    def cache_init(self, g, h, dropout):
        adj = g.adj().to_dense()

        # normalization
        degs = g.in_degrees().float()  # degrees of all nodes (1-dimension tensor)
        norm = torch.pow(degs, -0.5)  # 
        norm_tilde = torch.diag(norm) # 
        if self.cuda:
            norm_tilde.cuda()
        adj = torch.mm(torch.mm(norm_tilde, adj), norm_tilde)  # D^(-0/5) * A * D^(-0/5)
        
        # step2: aggregation
        cache_content = torch.mm(adj, h)

        return cache_content

    

