# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:21:51 2020

Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting
Refer to https://github.com/Davidham3/STSGCN

@author: Zhao Yiji
"""

import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from src.basic import DynamicAdj_dense, DynamicAdj_sparse, GSTAttentionBlock, GatedAttPrediction


def construct_adj(A, steps=3):
    '''
    construct a bigger adjacency matrix using the given matrix
    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)
    steps: how many times of the does the new adj mx bigger than A
    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    '''
    N = len(A)
    adj = np.zeros([N * steps] * 2)

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj

class STSGCN(nn.Block):
    
    def __init__(self, config, **kwargs):
        super(STSGCN, self).__init__(**kwargs)

        with self.name_scope():
            self.fc1 = nn.Dense(units=config.D, flatten=False, use_bias=True, activation=None)
            self.fc2 = nn.Dense(units=config.D, flatten=False, use_bias=True, activation=None)
            self.ln = nn.LayerNorm()

    def forward(self, x, adj):  
        # x:(B, T, t,V, D) adj:(B, T, tV, tV)
        B, T, t, V, D = x.shape
        
        # B,T,t,V,D -> BT,tV,D
        x = x.reshape((-3, t*V, D))
        
        # AX: BT,tV,tV x BT,tV,D -> BT,tV,D -> B,T,t,V,D
        x = nd.batch_dot(adj,x).reshape((B,T,t,V,D))
        # AXW: B,T,t,V,D x D,D
        x = self.fc1(x)*nd.sigmoid(self.fc2(x))
        # x = nd.relu(x+res)
        return self.ln(x)

class MutliLayerGCN(nn.Block):  # channel attention layer
    def __init__(self, config, num_stack=3):
        super(MutliLayerGCN, self).__init__()
        
        self.V = config.V
        self.D = config.D
        self.T = config.T
        
        with self.name_scope():
            self.gcn_lst = []
            for i in range(num_stack):
                self.gcn_lst.append(STSGCN(config=config))
                self.register_child(self.gcn_lst[-1])
    
    def forward(self, x, adj, adj_mask):        
        B,T,t,V,D = x.shape
        # adj: B,T,t,V,V-> B,T,V,t,V-> B,T,1,V,t,V
        adj = adj.transpose((0,1,3,2,4)).expand_dims(axis=2)
        adj = nd.repeat(adj,repeats=3,axis=2).reshape(B*T,t*V,t*V)
        adj = adj*adj_mask
        
        # x:B,T,t,V,D
        x_lst = []
        for gcn in self.gcn_lst:
            x = gcn(x, adj)
            x_lst.append(x[:,:,1])
        return nd.max(nd.stack(*x_lst,axis=0),axis=0)


class STBlock(nn.Block):
    def __init__(self, config, **kwargs):
        super(STBlock, self).__init__(**kwargs)

        self.config    = config
        self.do_att    = config.do_att
        self.G         = config.G
        self.D         = config.D
        self.V         = config.V
        
        self.t_size    = 3

        with self.name_scope():
            
            self.S_emb = self.params.get('S_emb', shape=(1, 1, self.V, self.D))
            # self.T_emb = self.params.get('T_emb', shape=(1, self.T, 1, self.D))
            self.T_emb = self.params.get('T_emb', allow_deferred_init=True)
            
            self.dgcn = MutliLayerGCN(config=config,num_stack=3)
            
            if self.do_att:
                self.att = GSTAttentionBlock(self.config)
                
    def forward(self, x, adj,adj_mask):
        
        # adj: B,T,V,V
        
        B,T,V,D = x.shape
        
        self.T_emb.shape = (1, T, 1, D)
        self.T_emb._finish_deferred_init()
        x = x+self.S_emb.data()+self.T_emb.data()

        x_lst   = [x[:,t:t+self.t_size] for t in range(T-self.t_size+1)] # [B,V,D]*T-1
        adj_lst = [adj[:,t:t+self.t_size] for t in range(T-self.t_size+1)] # [B,V,V]*T-1
        
        x = self.dgcn(nd.stack(*x_lst,axis=1),nd.stack(*adj_lst,axis=1),adj_mask)

        if self.do_att:
            x = self.att(x)

        return x
    
class STBlocks(nn.Block):
    def __init__(self, config, **kwargs):
        super(STBlocks, self).__init__(**kwargs)

        self.config    = config
        self.num_stack = config.num_stack
        self.T         = config.T

        with self.name_scope():
            self.st_lst  = []
            self.resfc_lst = []
            # self.ln_lst  = []
            for i in range(self.num_stack):
                self.st_lst.append(STBlock(self.config))
                self.register_child(self.st_lst[-1])
                
                self.resfc_lst.append(nn.Conv2D(channels=self.T-(i+1)*2, kernel_size=(1, 1),activation=None))
                self.register_child(self.resfc_lst[-1])
                
    def forward(self, x, adj,adj_mask):
        for i,(st,resfc) in enumerate(zip(self.st_lst,self.resfc_lst)):
            if i>0:
                adj = adj[:,1:-1]
            x = nd.relu(resfc(x) + st(x,adj,adj_mask))
        return x


class MR_STSGCN(nn.Block):
    def __init__(self, config,  **kwargs):
        super(MR_STSGCN, self).__init__(**kwargs)

        self.config = config
        self.G   = config.G
        self.T   = config.T
        self.V   = config.V
        self.D   = config.D
        self.dim_edge  = config.dim_edge
        self.num_stack = config.num_stack
        self.num_prediction = config.num_prediction
        
        self.dist = nd.array(config.dist, ctx=config.ctx).reshape(1,self.V,self.V)
        
        self.e_idx = config.spatial_edge_idx
        self.edge  = nd.array(config.spatial_edge, dtype='float32', ctx=config.ctx).expand_dims(axis=-1)
        self.E = self.edge.shape[0]

        self.tvtv_spatial = nd.array(construct_adj(config.gcn_spatial_adj), ctx=config.ctx)
        self.tvtv_traffic = nd.array(construct_adj(np.ones((self.V,self.V))), ctx=config.ctx)

        with self.name_scope():
           
            self.flow_enc_lst = []
            self.edge_enc_lst = []
            for i in range(self.G):
                self.flow_enc_lst.append(nn.Dense(units=self.D, flatten=False, use_bias=True, activation='relu'))
                self.register_child(self.flow_enc_lst[-1])
                self.edge_enc_lst.append(nn.Dense(units=self.dim_edge, flatten=False, use_bias=True, activation='relu'))
                self.register_child(self.edge_enc_lst[-1])
                
            # GCN adj   
            # I-I and O-O
            self.dyadj_i = DynamicAdj_sparse(config=self.config, edge_to_adj=self.e_idx, num_edge=self.E)
            self.dyadj_o = DynamicAdj_sparse(config=self.config, edge_to_adj=self.e_idx, num_edge=self.E)

            # I-O
            gcn_adj = (False, None)
            self.dyadj_io = DynamicAdj_dense(config=self.config, gcn_adj=gcn_adj)
            
            # ST
            self.st_lst = []
            for i in range(self.G):
                self.st_lst.append(STBlocks(self.config))
                self.register_child(self.st_lst[-1])
            
            self.predictor_i = GatedAttPrediction(config)
            self.predictor_o = GatedAttPrediction(config)


    def forward(self, x, adj_lst):

        B,T,V,D = x.shape
        
        # 1. x and xe
        x_lst = []
        xe_lst = []
        
        # I-I and O-O
        x_lst.append(x[:,:,:,0].expand_dims(axis=-1)) # x_in
        x_lst.append(x[:,:,:,1].expand_dims(axis=-1)) # x_out
        
        fea_1 = x[:,:,self.edge[:,0]] #B,T,E,1,D
        fea_2 = x[:,:,self.edge[:,1]] #B,T,E,1,D
        # E,D,1 -> 1,E,1 -> B,T,E,1
        fea_d = self.edge[:,2].expand_dims(axis=0).repeat(repeats=B*T,axis=0).reshape(B,T,self.E,1)
        
        xe_i = nd.concat(fea_1[:,:,:,:,0],fea_2[:,:,:,:,0],fea_d,adj_lst[0],dim=-1)
        xe_o = nd.concat(fea_1[:,:,:,:,1],fea_2[:,:,:,:,1],fea_d,adj_lst[0],dim=-1)
        
        xe_lst.append(xe_i)
        xe_lst.append(xe_o)
            
        # I-O
        x_lst.append(x)
        xe_lst.append(adj_lst[-1])
        
        # 2. x and xe encoding
        x_lst  = [enc(x) for x,enc  in zip(x_lst, self.flow_enc_lst)]
        xe_lst = [enc(xe) for xe,enc in zip(xe_lst,self.edge_enc_lst)]
        
        # 3. dynamic edge weight
        adj_lst = [self.dyadj_i(xe_lst[0]), self.dyadj_o(xe_lst[1]), self.dyadj_io(xe_lst[2])]

        adj_mask_lst = [self.tvtv_spatial,self.tvtv_spatial,self.tvtv_traffic]

        # 4. Spatio-Temporal
        x_lst  = [st(x,adj,adj_mask)  for x,adj,adj_mask,st  in zip(x_lst, adj_lst,adj_mask_lst, self.st_lst)]

        # 5.Pprediction
        
        x_i = self.predictor_i(x_lst[0],x_lst[2])
        x_o = self.predictor_o(x_lst[1],x_lst[2])

            
        # [B,Tp,V,1]*2 -> B,Tp,V,2
        return nd.concat(x_i, x_o, dim=-1)