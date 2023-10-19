# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:21:51 2020

Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
Refer to https://github.com/Davidham3/STGCN

@author: Zhao Yiji
"""

from mxnet import nd
from mxnet.gluon import nn
from src.basic import GCN, DynamicAdj_dense, DynamicAdj_sparse, GSTAttentionBlock, GatedAttPrediction

class STBlock(nn.Block):
    def __init__(self, config, **kwargs):
        super(STBlock, self).__init__(**kwargs)

        self.config    = config
        self.do_att    = config.do_att
        self.G         = config.G
        self.D         = config.D

        with self.name_scope():
           
            self.gcn = GCN(config=self.config)
            self.tconv11 = nn.Conv2D(channels = self.D, kernel_size = (3, 1), padding = (1, 0), strides = (1, 1), activation=None, layout='NHWC')
            self.tconv12 = nn.Conv2D(channels = self.D, kernel_size = (3, 1), padding = (1, 0), strides = (1, 1), activation='sigmoid', layout='NHWC')
            self.tconv21 = nn.Conv2D(channels = self.D, kernel_size = (3, 1), padding = (1, 0), strides = (1, 1), activation=None, layout='NHWC')
            self.tconv22 = nn.Conv2D(channels = self.D, kernel_size = (3, 1), padding = (1, 0), strides = (1, 1), activation='sigmoid', layout='NHWC')

            if self.do_att:
                self.att = GSTAttentionBlock(self.config)
            
    def forward(self, x, adj):

        x = self.tconv11(x)*self.tconv12(x)
        x = self.gcn(x,adj)
        x = self.tconv21(x)*self.tconv22(x)

        if self.do_att:
            x = self.att(x)
        return x

class STBlocks(nn.Block):
    def __init__(self, config, **kwargs):
        super(STBlocks, self).__init__(**kwargs)

        self.config    = config
        self.num_stack = config.num_stack

        with self.name_scope():
            self.st_lst = []
            for i in range(self.num_stack):
                self.st_lst.append(STBlock(self.config))
                self.register_child(self.st_lst[-1])
                
    def forward(self, x, adj):
        for st in self.st_lst:
            x = x + st(x,adj) #残差
        return x

class MR_STGCN(nn.Block):
    def __init__(self, config,  **kwargs):
        super(MR_STGCN, self).__init__(**kwargs)

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

        # 4. Spatio-Temporal
        x_lst  = [st(x,adj)  for x,adj,st  in zip(x_lst, adj_lst, self.st_lst)]
        
        # 5.Pprediction
        x_i = self.predictor_i(x_lst[0],x_lst[2])
        x_o = self.predictor_o(x_lst[1],x_lst[2])

            
        # [B,Tp,V,1]*2 -> B,Tp,V,2
        return nd.concat(x_i, x_o, dim=-1)
    
