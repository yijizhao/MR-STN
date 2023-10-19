# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:21:51 2020

Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting (ASTGCN)
Refer to https://github.com/Davidham3/ASTGCN

@author: Zhao Yiji
"""

from mxnet import nd
from mxnet.gluon import nn
from src.basic import GCN, DynamicAdj_dense, DynamicAdj_sparse, GSTAttentionBlock, GatedAttPrediction

class Spatial_Attention_layer(nn.Block):
    '''
    compute spatial attention scores
    '''
    def __init__(self, **kwargs):
        super(Spatial_Attention_layer, self).__init__(**kwargs)
        with self.name_scope():
            self.W_1 = self.params.get('W_1', allow_deferred_init=True)
            self.W_2 = self.params.get('W_2', allow_deferred_init=True)
            self.W_3 = self.params.get('W_3', allow_deferred_init=True)
            self.b_s = self.params.get('b_s', allow_deferred_init=True)
            self.V_s = self.params.get('V_s', allow_deferred_init=True)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})
        Returns
        ----------
        S_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, N, N)
        '''
        # x: B,V,F,T
        # B,T,V,F->B,V,F,T
        # x = x.transpose((0,2,3,1))
        
        # get shape of input matrix x
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # defer the shape of params
        self.W_1.shape = (num_of_timesteps, )
        self.W_2.shape = (num_of_features, num_of_timesteps)
        self.W_3.shape = (num_of_features, )
        self.b_s.shape = (1, num_of_vertices, num_of_vertices)
        self.V_s.shape = (num_of_vertices, num_of_vertices)
        for param in [self.W_1, self.W_2, self.W_3, self.b_s, self.V_s]:
            param._finish_deferred_init()

        # compute spatial attention scores
        # shape of lhs is (batch_size, V, T)
        lhs = nd.dot(nd.dot(x, self.W_1.data()), self.W_2.data())

        # shape of rhs is (batch_size, T, V)
        rhs = nd.dot(self.W_3.data(), x.transpose((2, 0, 3, 1)))

        # shape of product is (batch_size, V, V)
        product = nd.batch_dot(lhs, rhs)

        S = nd.dot(self.V_s.data(),
                   nd.sigmoid(product + self.b_s.data())
                     .transpose((1, 2, 0))).transpose((2, 0, 1))

        # normalization
        S = S - nd.max(S, axis=1, keepdims=True)
        exp = nd.exp(S)
        S_normalized = exp / nd.sum(exp, axis=1, keepdims=True)
        return S_normalized

class Temporal_Attention_layer(nn.Block):
    '''
    compute temporal attention scores
    '''
    def __init__(self, **kwargs):
        super(Temporal_Attention_layer, self).__init__(**kwargs)
        with self.name_scope():
            self.U_1 = self.params.get('U_1', allow_deferred_init=True)
            self.U_2 = self.params.get('U_2', allow_deferred_init=True)
            self.U_3 = self.params.get('U_3', allow_deferred_init=True)
            self.b_e = self.params.get('b_e', allow_deferred_init=True)
            self.V_e = self.params.get('V_e', allow_deferred_init=True)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h
                       shape is (batch_size, N, C_{r-1}, T_{r-1})
        Returns
        ----------
        E_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})
        '''
        
        # x: B,V,F,T
        # B,T,V,F->B,V,F,T
        x = x.transpose((0,2,3,1))
        
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # defer shape
        self.U_1.shape = (num_of_vertices, )
        self.U_2.shape = (num_of_features, num_of_vertices)
        self.U_3.shape = (num_of_features, )
        self.b_e.shape = (1, num_of_timesteps, num_of_timesteps)
        self.V_e.shape = (num_of_timesteps, num_of_timesteps)
        for param in [self.U_1, self.U_2, self.U_3, self.b_e, self.V_e]:
            param._finish_deferred_init()

        # compute temporal attention scores
        # shape is (N, T, V)
        lhs = nd.dot(nd.dot(x.transpose((0, 3, 2, 1)), self.U_1.data()),
                     self.U_2.data())

        # shape is (N, V, T)
        rhs = nd.dot(self.U_3.data(), x.transpose((2, 0, 1, 3)))

        product = nd.batch_dot(lhs, rhs)

        E = nd.dot(self.V_e.data(),
                   nd.sigmoid(product + self.b_e.data())
                     .transpose((1, 2, 0))).transpose((2, 0, 1))

        # normailzation
        E = E - nd.max(E, axis=1, keepdims=True)
        exp = nd.exp(E)
        E_normalized = exp / nd.sum(exp, axis=1, keepdims=True)
        return E_normalized


class STBlock(nn.Block):
    def __init__(self, config, **kwargs):
        super(STBlock, self).__init__(**kwargs)

        self.config    = config
        self.do_att    = config.do_att
        self.G         = config.G
        self.D         = config.D

        with self.name_scope():
            
            self.SAt = Spatial_Attention_layer()
            self.TAt = Temporal_Attention_layer()
            
            self.gcn = GCN(config=self.config)
            self.tconv = nn.Conv2D(channels = self.D, kernel_size = (3, 1), padding = (1, 0), strides = (1, 1), activation=None, layout='NHWC')
            self.residual_conv = nn.Conv2D(channels=self.D,kernel_size=(1, 1),strides=(1, 1),activation=None, layout='NHWC')

            if self.do_att:
                self.att = GSTAttentionBlock(self.config)
                
            self.ln = nn.LayerNorm()
            
    def forward(self, x, adj):
        
        B,T,V,F = x.shape
        
        res = self.residual_conv(x)
        
        temporal_At = self.TAt(x)
        x_TAt = nd.batch_dot(x.transpose((0,2,3,1)).reshape(B, -1, T),temporal_At).reshape(B,V,F,T)
        spatial_At = self.SAt(x_TAt).expand_dims(axis=1)
        
        x = self.gcn(x,adj*spatial_At)
        x = self.tconv(x)
        x = self.ln(nd.relu(x + res))

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

class MR_ASTGCN(nn.Block):
    def __init__(self, config,  **kwargs):
        super(MR_ASTGCN, self).__init__(**kwargs)

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
            
        # O-O
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