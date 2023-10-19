# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:21:51 2020

@author: Zhao Yiji
"""

from mxnet import nd
from mxnet.gluon import nn,rnn
from src.basic import GCN, DynamicAdj_dense, DynamicAdj_sparse, GSTAttentionBlock, GatedAttPrediction


class MutliLayerGCN(nn.Block):
    def __init__(self, config, num_stack=2):
        super(MutliLayerGCN, self).__init__()
        with self.name_scope():
            self.gcn_lst = []
            for i in range(num_stack):
                self.gcn_lst.append(GCN(config=config))
                self.register_child(self.gcn_lst[-1])
    
    def forward(self, x, adj):
        for gcn in self.gcn_lst:
            x = gcn(x,adj)
        return x
    
class Temporal_Gate_Layer(nn.Block):
    def __init__(self, config):
        super(Temporal_Gate_Layer, self).__init__()
        with self.name_scope():
            self.W_1 = self.params.get('W_1', allow_deferred_init=True)
            self.W_2 = self.params.get('W_2', allow_deferred_init=True)
            self.dgcn = MutliLayerGCN(config=config,num_stack=2)
            self.ln = nn.LayerNorm()
            
    def forward(self, x, adj):
        '''
        :param x: B,T,V,F
        :return: B,T,1,1
        '''
        B, T, V, F  = x.shape
        self.W_1.shape = (T, T)
        self.W_1._finish_deferred_init()
        self.W_2.shape = (T, T)
        self.W_2._finish_deferred_init()
        
        spatial_gcn = self.dgcn(x,adj)
        x_ = nd.concat(x, spatial_gcn, dim=2)
        x_ = self.ln(x_)
        # B,T,V,F -> B,V,F,T
        z = nd.mean(x_, axis=(2, 3)) #B,T
        s = nd.sigmoid(nd.dot(nd.relu(nd.dot(z, self.W_1.data())), self.W_2.data()))  #B, T

        return x*s.reshape((B,T,1,1))

class CGRNN(nn.Block):
    def __init__(self, config):
        super(CGRNN, self).__init__()
        self.D = config.D
        self.TAtt = Temporal_Gate_Layer(config)
        self.rnn  = rnn.RNN(hidden_size=self.D, num_layers=3)
        self.dgcn = MutliLayerGCN(config=config,num_stack=5)
        self.ln = nn.LayerNorm()

    def forward(self, x, adj):
        B,T,V,D = x.shape
        x = self.TAtt(x, adj).transpose((1,0,2,3)).reshape(T,-3,D)  # B,T,V,D->T,B,V,D->T,BV,D
        x = self.rnn(x)
        x = x.reshape(T,B,V,D).transpose((1,0,2,3))
        x = self.dgcn(x,adj)
        return self.ln(x)


class STBlock(nn.Block):
    def __init__(self, config, **kwargs):
        super(STBlock, self).__init__(**kwargs)

        self.config    = config
        self.do_att    = config.do_att
        self.G         = config.G
        self.D         = config.D

        with self.name_scope():
            self.cgrnn = CGRNN(config=self.config)
            
            if self.do_att:
                self.att = GSTAttentionBlock(self.config)
                
    def forward(self, x, adj):
        
        B,T,V,F = x.shape
        x = self.cgrnn(x,adj)
        
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

class MR_STMGCN(nn.Block):
    def __init__(self, config,  **kwargs):
        super(MR_STMGCN, self).__init__(**kwargs)

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