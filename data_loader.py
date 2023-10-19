# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:21:51 2020

@author: Zhao Yiji
"""

import numpy as np
from mxnet import gluon
from sklearn.preprocessing import StandardScaler

np_type = 'float16'
nd_type = 'float32'

def search_recent_data(train, label_start_idx, points_per_hour, num_prediction):
    if label_start_idx + num_prediction > len(train): return None
    start_idx, end_idx = label_start_idx - num_prediction, label_start_idx - num_prediction + num_prediction    
    if start_idx < 0 or end_idx < 0: return None
    return (start_idx, end_idx), (label_start_idx, label_start_idx + num_prediction)
        
class CleanDataset():
    def __init__(self, config):
        
        self.config = config
        self.feature_file   = config.data.feature_file
        self.adj_file_lst   = config.data.gfile_lst
        self.val_start_idx  = config.data.val_start_idx
        
        self.data_name = config.data.name

        self.feature = self._normalization().astype(np_type) #(T, V, D)
        self.labels  = np.load(self.feature_file).astype(np_type)
           
        self.speed   = self.gen_speed_sparse().astype(np_type)
        self.transit = np.expand_dims(self._get_normalized_adj(self.adj_file_lst[1]),axis=-1).astype(np_type)
        
    def gen_speed_sparse(self):
        '''
            speed:<T, V, V, 1>
            return: e_idx_arr and E, shape: (E, T, 1)
        '''
        
        speed = np.expand_dims(self._get_normalized_adj(self.adj_file_lst[0]),axis=-1)
        adj   = np.load(self.config.data.path+self.config.data.name+'/adj_spatial.npy')
        
        E = []
        e_idx_arr = np.zeros(adj.shape)
        idx = np.where(adj > 0)
        e_idx_arr[idx] = np.arange(len(idx[0])) + 1
        for i_idx in range(adj.shape[0]):
            for j_idx in range(adj.shape[1]):
                if adj[i_idx, j_idx] > 0:
                    E.append(speed[:, i_idx, j_idx]) 
        return np.array(E).transpose((1,0,2))
        
        
    def _get_normalized_adj(self, adj_file):
        adj = np.load(adj_file)
        
        if len(adj.shape)==2:#static graph (V, V)
            print('static graph ->', adj_file)
            adj = self._normalized_adj(adj)
            
        elif len(adj.shape)>2:#dynamic graph (T, V, V) or (T, V, V, D)
            print('dynamic graph ->', adj_file)
            adj = self._normalization_edge(adj)

        return adj

    def _normalized_adj(self, adj):
        adj = adj + np.eye(adj.shape[0])
        degree = adj.sum(axis=1)
        degree = np.diagflat(np.power(degree, -0.5))
        adj = adj.dot(degree).transpose().dot(degree) #(AD^-0.5)^T * D^-0.5 = D^-0.5AD^-0.5
        adj[np.isinf(adj)] = 0.
        return adj
      
    def _normalization(self):
        '''
        对数据进行标准化
        '''
        feature = np.load(self.feature_file)
        train = feature[:self.val_start_idx]
        if self.data_name == 'Metro': # used 6:00-24:00
            idx_lst = [i for i in range(train.shape[0]) if i % (24*6) >= 7*6 - 12]
            train = train[idx_lst]

        transformer = StandardScaler().fit(train.reshape(train.shape[0], -1))
        feature = transformer.transform(feature.reshape(feature.shape[0], -1)).reshape(feature.shape) #(T, V, D)
        
        return feature

    def _normalization_edge(self, adj):

        #T,V,V = adj.shape
        adj_shape = adj.shape
        train = adj[:self.val_start_idx]
        if self.data_name == 'Metro':
            idx_lst = [i for i in range(train.shape[0]) if i % (24*6) >= 7*6 - 12]
            train = train[idx_lst]

        transformer = StandardScaler().fit(train.reshape(train.shape[0], -1))
        adj = transformer.transform(adj.reshape(adj_shape[0], -1)) #(T,VVF)or (T,VV)
        adj   = adj.reshape(adj_shape)
        
        return adj.astype(np_type)


class TrafficDataset(gluon.data.Dataset):

    def __init__(self, clean_data, data_range, config):
        
        self.T = config.model.T
        self.V = config.model.V
        self.adj = config.model.gcn_spatial_adj
        self.adj_arg = np.argwhere(self.adj>0)
        
        self.data_path  = config.data.path+config.data.name+'/'
        self.data_range = data_range
        
        self.data_name = clean_data.data_name
        self.labels    = clean_data.labels
        self.feature   = clean_data.feature #(T, V, D)
        
        self.speed     = clean_data.speed
        self.transit   = clean_data.transit
        
        
        self.num_recent = config.data.num_recent
        self.points_per_hour = config.data.points_per_hour
        self.num_prediction  = config.model.num_prediction

        # Splitting the dataset
        self.idx_lst = self.get_idx_lst()
        print('samples:',len(self.idx_lst))
          
    
    def __getitem__(self, index):
        # (T, V, D)
        idx  = self.idx_lst[index] # 0: idx list，1: start and end

        start,end = idx[1][0],idx[1][1]
        label = self.labels[start:end]

        start,end = idx[0][0],idx[0][1]
        node_feature     = self.feature[start:end]
        edge_feature_lst = [self.speed[start:end],self.transit[start:end]]
        
        return (label, node_feature) + tuple(edge_feature_lst)        
    
    def __len__(self):
        return len(self.idx_lst)

    def get_idx_lst(self):
        
        idx_lst = []
        start = self.data_range[0]
        end   = self.data_range[1] if self.data_range[1]!=-1 else self.feature.shape[0]
        
        for label_start_idx in range(start,end):
            
            if self.data_name == 'Metro': # used 6:00-24:00
                if label_start_idx % (24 * 6) < (7*6):
                    continue
                if label_start_idx % (24 * 6) > (24*6) - self.num_prediction:
                    continue            

            recent = search_recent_data(self.feature, label_start_idx, self.points_per_hour, self.num_prediction)
            if recent:
                idx_lst.append(recent)                 
        return idx_lst 