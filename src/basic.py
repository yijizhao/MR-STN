# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:21:51 2020

@author: Zhao Yiji
"""

from mxnet import nd
from mxnet.gluon import nn

class GCN(nn.Block):
    def __init__(self, config, **kwargs):
        super(GCN, self).__init__(**kwargs)
        
        self.gcn_dim  = config.D
        with self.name_scope():
            self.fc = nn.Dense(units=self.gcn_dim, flatten=False, use_bias=True, activation=None)

    def forward(self, x, adj):  
        # x:(B, T, V, D) adj:(B,T, V, V)
        B, T, V, D = x.shape

        # B,T,V,D -> BT,V,D
        x   = x.reshape((-3, V, D))  
        #BT,V,V
        adj = adj.reshape((-3, V, V))
        # AX: BT,V,V x BT,V,D -> BT,V,D -> B,T,V,D
        x = nd.batch_dot(adj,x).reshape((B,T,V,D))
        # AXW: B,T,V,D x D,D
        x = self.fc(x)
        x = nd.relu(x)

        return x

class DynamicAdj_dense(nn.Block):
    
    def __init__(self, config, gcn_adj=None, **kwargs):
        super(DynamicAdj_dense, self).__init__(**kwargs)
                
        self.do_adj_mask   = gcn_adj[0]
        if self.do_adj_mask:
            print(gcn_adj[1][0][0])
            self.mask_adj = nd.sign(gcn_adj[1])
        
        with self.name_scope():
            self.M = self.params.get('MaskW', allow_deferred_init = True)
            self.B = self.params.get('MaskB', allow_deferred_init = True)

    def forward(self, adj):
        # adj: (B, T, V, V, D)
        B, T, V, V, D = adj.shape

        self.M.shape = (V, V, D)
        self.M._finish_deferred_init()
        self.B.shape = (V, V)
        self.B._finish_deferred_init()
        
        # B,T,V,V,D * V,V,D -> B,T,V,V + V,V -> B,T,V,V
        adj = nd.broadcast_mul(adj, self.M.data())
        adj = nd.sum_axis(adj,axis=-1)
        adj = nd.broadcast_add(adj, self.B.data())
        
        #B,T,V,V
        adj = nd.sigmoid(adj)
                
        if self.do_adj_mask:
            adj = nd.broadcast_mul(adj, self.mask_adj)
        return adj

class DynamicAdj_sparse(nn.Block):
    
    def __init__(self, config, edge_to_adj, num_edge,  **kwargs):
        super(DynamicAdj_sparse, self).__init__(**kwargs)
                
        self.ctx = config.ctx
        self.edge_idx = edge_to_adj
        self.E = num_edge
        self.V = config.V

        with self.name_scope():
            self.w = self.params.get('MaskW', allow_deferred_init = True)
            self.b = self.params.get('MaskB', allow_deferred_init = True)

    def forward(self, adj):
        # adj: (B,T,E,D)  
        
        #E:B,T,E,D
        B,T,E,D = adj.shape
        
        self.w.shape = (E, D, 1)
        self.w._finish_deferred_init()
        self.b.shape = (E, 1, 1)
        self.b._finish_deferred_init()
        
        # B,T,E,D-> E,(BT,D x D,1)-> E,BT,1 + E,1,1
        adj = nd.batch_dot(adj.transpose((2,0,1,3)).reshape(0,-3,0), self.w.data()) + self.b.data()
        #(E+1,BT,1)->V,V,BT,1
        adj = nd.concat(nd.zeros((1, B*T, 1),ctx=self.ctx),nd.sigmoid(adj),dim=0)
        
        return adj[self.edge_idx].transpose((2,0,1,3)).reshape(B,T,self.V,self.V)

class FFN(nn.Block):
    def __init__(self, in_dim, out_dim, drop=0.0, **kwargs):
        super(FFN, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1  = nn.Dense(in_units=in_dim, units=in_dim, flatten=False, prefix='fc1_')
            self.fc2  = nn.Dense(in_units=in_dim, units=out_dim, flatten=False, prefix='fc2_')
        self._dropout_layer_ffn = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = nd.relu(x)
        x = self._dropout_layer_ffn(x)
        x = self.fc2(x)
        return x    

class GSTAttentionBlock(nn.Block):   
    def __init__(self, config, **kwargs):
        super(GSTAttentionBlock, self).__init__(**kwargs)

        self.D = config.D
        self.gather_dim = config.att_gather_dim
        self.drop    = config.dropout

        with self.name_scope():
            self.Wq = self.params.get('Wq', allow_deferred_init = True)
            self.Wk = self.params.get('Wk', allow_deferred_init = True)
            self.Wv = self.params.get('Wv', allow_deferred_init = True)

            self.FFN = FFN(self.D, self.D, drop=self.drop)
            self.register_child(self.FFN)
            
            self.ln = nn.LayerNorm()
    
    def forward(self, x):

        B, T, V, D = x.shape
        
        # defer the shape of params
        params_lst = []
        self.Wq.shape = (D, self.gather_dim)
        self.Wk.shape = (D, self.gather_dim)
        self.Wv.shape = (D, D)
        params_lst.extend([self.Wq, self.Wk, self.Wv])

        for param in params_lst:
            param._finish_deferred_init()

        # (B,T,V,D) -> (B,TV,D)
        x = x.reshape((B, -3, D))
        
        # shape of Q,K,V is (B,TV,D)
        query = nd.dot(x, self.Wq.data()) #(B,TV,D')
        key   = nd.dot(x, self.Wk.data()) #(B,TV,D')
        value = nd.dot(x, self.Wv.data()) #(B,TV,D)

        #Q*(KV)
        query = nd.softmax(query, axis=2)
        key   = nd.softmax(key,  axis=1)
        
        # K.*V: B,TV,D' -> B,D',TV x B,TV,D -> B,D',D
        context = nd.batch_dot(key, value, transpose_a=True)

        # Q.*KV: B,TV,D' x B,D',D ->  B, TV, D
        context = nd.batch_dot(query, context)
        
        return self.FFN(self.ln(context+x)).reshape((B, T, V, D))


class AddAttention(nn.Block):
    def __init__(self, hidden_dim, num_prediction):
        super(AddAttention, self).__init__()
        
        self.P = num_prediction

        with self.name_scope():

            self.w = self.params.get('att_W', shape=(hidden_dim, hidden_dim,self.P))
            self.b = self.params.get('att_b', shape=(hidden_dim, self.P))
            self.q = self.params.get('att_q', shape=(self.P, hidden_dim, 1))

    def forward(self, x): 
        # B,T,V,D x D,D,P + D,P -> B,T,V,D,P
        # P,(B,T,V,D x D,1) -> P,B,T,V,1(Att)
        # P,B,T,V,1 * B,T,V,D 然后sum(axis=1)合并T
        # v^T * tanh(wx+b)
        
        B,T,V,D = x.shape
        
        u = nd.tanh(nd.dot(x,self.w.data())+self.b.data())
        c = nd.batch_dot(u.transpose((4,0,1,2,3)).reshape(self.P,B*T*V,D), self.q.data())
        a = nd.softmax(c.reshape(self.P,B,T,V,1), axis=2)
        return (a * x).sum(axis=2) #P,B,V,D


class PredFFN(nn.Block):
    def __init__(self, num_prediction, in_dim, drop=0.0, **kwargs):
        super(PredFFN, self).__init__(**kwargs)

        self.P = num_prediction

        self.fcW_1 = self.params.get('fcW_1', shape=(self.P, in_dim, in_dim))
        self.fcb_1 = self.params.get('fcb_1', shape=(self.P, 1, in_dim))
        self.fcW_2 = self.params.get('fcW_2', shape=(self.P, in_dim, 1))
        self.fcb_2 = self.params.get('fcb_2', shape=(self.P, 1, 1))

        self.dropout_layer = nn.Dropout(drop)
        
    def forward(self, x):
        # x: P,B,V,D
        # P,(BV,D x D,D + D)
        P,B,V,D = x.shape
        x = nd.batch_dot(x.reshape(0,-3,0), self.fcW_1.data()) + self.fcb_1.data()
        x = nd.relu(x)
        x = self.dropout_layer(x)
        x = nd.batch_dot(x, self.fcW_2.data()) + self.fcb_2.data()
        x = x.reshape(self.P,B,V,1).transpose((1,0,2,3))
        return x


class GatedAttPrediction(nn.Block):
    def __init__(self, config):
        super(GatedAttPrediction, self).__init__()
        self.D    = config.D
        self.drop = config.dropout
        self.num_prediction = config.num_prediction

        with self.name_scope():
            self.gw1 = nn.Dense(units=self.D, flatten=False, use_bias=True, activation=None)
            self.gw2 = nn.Dense(units=self.D, flatten=False, use_bias=False, activation=None)
            self.att = AddAttention(hidden_dim=self.D,num_prediction=self.num_prediction)
            self.ffn = PredFFN(num_prediction=self.num_prediction, in_dim=self.D, drop=self.drop)
            self.ln = nn.LayerNorm()
                
    def forward(self, x1, x2): 
        # B,T,V,D * G
        g = nd.sigmoid(self.gw1(x1)+self.gw2(x2))
        x = g*x1 + (1-g)*x2
        
        x = self.ffn(self.ln(self.att(x)))
        
        return x
    
class AttPrediction(nn.Block):
    def __init__(self, config):
        super(AttPrediction, self).__init__()
        self.D    = config.D
        self.drop = config.dropout
        self.num_prediction = config.num_prediction

        with self.name_scope():
            self.att = AddAttention(hidden_dim=self.D,num_prediction=self.num_prediction)
            self.ffn = PredFFN(num_prediction=self.num_prediction, in_dim=self.D, drop=self.drop)
            self.ln = nn.LayerNorm()
                
    def forward(self, x): 
        x = self.ffn(self.ln(self.att(x)))
        return x

