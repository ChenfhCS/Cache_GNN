import os
import sys 

import numpy as np
import numba
import torch
from dgl import DGLGraph
# from dgl.frame import Frame, FrameRef
import dgl.utils

class GraphCacheServer:
    def __init__(self, graph, node_num, device):
        self.graph = graph  # dgl graph
        self.device = device  # device
        self.node_num = node_num  # number of nodes in the graph

        print(self.graph.ndata['feat'])

        # self.graph.ndata['features'] = self.graph.ndata.pop('feat')
        # self.graph.ndata['labels'] = self.graph.ndata.pop('label')

        # masks for manage the feature locations: default in CPU
        self.gpu_flag = torch.zeros(self.node_num).bool().cuda(self.device)
        self.gpu_flag.requires_grad_(False)

        self.full_cached = False

        self.gpu_cache = dict()

        # id map from local id to cache id
        with torch.cuda.device(self.device):
            self.IdMap_local_cache = torch.cuda.LongTensor(node_num).fill_(0)  # GPU tensor filled 0: tensor[0,0,0,...,0]
            self.IdMap_local_cache.requires_grad_(False)


    def cache_init(self, embed_names):
        # Step1: get available GPU memory
        peak_allocated_mem = torch.cuda.max_memory_allocated(device=self.device)
        peak_cached_mem = torch.cuda.max_memory_cached(device=self.device)
        total_mem = torch.cuda.get_device_properties(self.device).total_memory
        available = total_mem - peak_allocated_mem - peak_cached_mem \
                    - 1024 * 1024 * 1024 # in bytes
        
        # Stpe2: get capability
        # self.capability = int(available / (self.total_dim * 4)) 
        self.capability = 20 # at first, we set the capability manually

        # Step3: cache features
        if self.capability >= self.node_num:
            # fully cache
            print('cache the full graph...')
            full_nids = torch.arange(self.node_num).cuda(self.device)
            data_frame = self.get_features(full_nids, embed_names)
            self.cache_data(full_nids, data_frame, is_full=True)
        
        else:
            # choose top-capability out-degree nodes to cache
            print('cache the part of graph. Caching percentage: {:.4f}'
                    .format(self.capability / self.node_num))
            out_degrees = self.graph.out_degrees()  # get nodes degrees
            sort_nid = torch.argsort(out_degrees, descending=True)  # sort nodes with degree decreasing
            cache_nid = sort_nid[:self.capability]  # choose first capability nodes with the most degrees
            embed_dict = self.get_features(cache_nid, embed_names) # obtain the features of the chosen nodes
            self.cache_data(cache_nid, embed_dict, is_full=False)  # cache features
    

    def get_features(self, nids, embed_names, to_gpu=False):
        ''' 
        Get features from CPU, embed_names = ['features', 'norm']?
        nids: index of choen nodes for cache
        '''
        if to_gpu:
            embed_dict = {name: self.graph.ndata[name][nids].cuda(self.device, non_blocking=True)\
                for name in embed_names}
        else:
            embed_dict = {name: self.graph.ndata[name][nids] for name in embed_names}
        return embed_dict

    def cache_data(self, nids, data, is_full=False):
        num = nids.size(0)  # number of chosen nodes
        '''
        id map from local idx to cache idx.
        For example, cache_id = [0,3,2,5,1], total 6 nodes,
        IdMap_local_cache = ['0', '4', '2', '1', 0, '3'];
        IdMap_local_cache[local_id] = cache_id.
        '''
        self.IdMap_local_cache[nids] = torch.arange(num).cuda(self.device)
        self.cached_num = num

        for name in data:
            self.gpu_cache[name] = data[name].cuda(self.device)
        
        # setup flags
        self.gpu_flag[nids] = True
        self.full_cached = is_full

    def fetch_data(self):
        return 
