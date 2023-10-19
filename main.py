# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:21:51 2020

@author: Zhao Yiji
"""

import argparse
import os
import math
import random
import numpy as np
from datetime import datetime
from easydict import EasyDict as edict
from timeit import default_timer as timer

from mxnet import nd
from mxnet import gpu
from mxnet import cpu
from mxnet import init
from mxnet import gluon
from mxnet import random as mxrandom
from mxnet import autograd

from utils import Metric, Logger
from data_loader import CleanDataset, TrafficDataset
from src.mr_stgcn  import MR_STGCN
from src.mr_astgcn import MR_ASTGCN
from src.mr_stmgcn import MR_STMGCN
from src.mr_stsgcn import MR_STSGCN

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
os.environ["MXNET_CUDA_LIB_CHECKING"] = "0"

np_type = 'float16'
nd_type = 'float32'

def train(model, data_loader, trainer, loss_function, epoch, metric, config):

    y_pred, y_true, time_lst = [],[],[]
    for i, feed in enumerate(data_loader):
        target           = nd.array(feed[0], ctx=config.ctx, dtype=nd_type)
        node_feature     = nd.array(feed[1], ctx=config.ctx, dtype=nd_type)
        edge_feature_lst = [nd.array(f, ctx=config.ctx, dtype=nd_type) for f in feed[2:]]        
        
        time_start = timer()
        with autograd.record():
            output = model(node_feature, edge_feature_lst)
            l = loss_function(output, target)
        l.backward(retain_graph=True)
        trainer.step(target.shape[0])
        time_lst.append((timer() - time_start))
                        
        if i == 0 and epoch==0:
            # --------------
            num_of_parameters = 0
            for param_name, param_value in model.collect_params().items():
                num_of_parameters += np.prod(param_value.shape)
            print('num_of_parameters', num_of_parameters)
            # --------------        
        
        y_true.append(target.asnumpy())
        y_pred.append(output.asnumpy())
                
        message = f"{i/len(data_loader)+epoch:6.1f} Time:{np.sum(time_lst):.1f}s"
        print('\r'+message , end='', flush=True)

    y_true = np.concatenate(y_true,axis=0)
    y_pred = np.concatenate(y_pred,axis=0)
    
    time_cost = np.sum(time_lst)
    metric.update_metrics(y_true,y_pred)
    metric.update_best_metrics(epoch=epoch)
    
    message = f"{epoch+1:<3} | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}{metric.metrics['mape']:<7.2f}{time_cost:<5.1f}s"
    print('\r'+message , end='', flush=False)
    
    message = f"{'Train':5}{epoch+1:6.1f} | {str(metric)}  {time_cost:.1f}s"
    config.logger.write('\n'+message+'\n',is_terminal=False)
    
    return metric

def evals(model, data_loader, epoch, metric, config, mode='Tes', end=''):
    
    y_pred, y_true, time_lst = [],[],[]
    for i, feed in enumerate(data_loader):
        
        target           = nd.array(feed[0], ctx=config.ctx, dtype=nd_type)
        node_feature     = nd.array(feed[1], ctx=config.ctx, dtype=nd_type)
        edge_feature_lst = [nd.array(f, ctx=config.ctx, dtype=nd_type) for f in feed[2:]]
        
        time_start = timer()
        output = model(node_feature, edge_feature_lst)
        time_lst.append((timer() - time_start))
        
        y_true.append(target.asnumpy())
        y_pred.append(output.asnumpy())
        
    y_true = np.concatenate(y_true,axis=0)
    y_pred = np.concatenate(y_pred,axis=0)    
        
    time_cost = np.sum(time_lst)
    metric.update_metrics(y_true,y_pred)#总
    metric.update_metrics_in_out(y_true,y_pred)#分
    metric.update_best_metrics(epoch=epoch)

    if mode == 'Val':
        message = f" | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}{metric.best_metrics['epoch']+1:<3}" 
    else:
        message = f" | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}{metric.best_metrics['epoch']+1:<4} {time_cost:<3.1f}s |{metric.metrics['time']}\n"
    
    print(message , end=end, flush=False)
    message = f"{mode:5}{epoch+1:6.1f} | {str(metric)}  {time_cost:.1f}s"
    config.logger.write(message+'\n',is_terminal=False)

    if mode != 'Val':
        metric.update_step_metrics(y_true,y_pred,epoch=epoch)#总
        
        for i,m in enumerate(metric.log_lst(epoch=epoch,sep=',')):
            if i%6==0:
                config.logger.write(('-'*20)+'\n',is_terminal=False)
            config.logger.write(m+'\n',is_terminal=False)
            
        # if metric.best_metrics['epoch'] == epoch:
        # np.save(config.fprediction,y_pred)
        
    return metric

class MyInit(init.Initializer):
    xavier = init.Xavier()
    uniform = init.Uniform()
    def _init_weight(self, name, data):
        if len(data.shape) < 2:
            self.uniform._init_weight(name, data)
            print('Init', name, data.shape, 'with Uniform')
        else:
            self.xavier._init_weight(name, data)
            print('Init', name, data.shape, 'with Xavier')

def main(config):

    # Model
    print('3. creat model ...')
    if config.model.mode == 'stgcn':
        model = MR_STGCN(config=config.model)
    elif config.model.mode == 'astgcn':
        model = MR_ASTGCN(config=config.model)
    elif config.model.mode == 'stmgcn':
        model = MR_STMGCN(config=config.model)
    elif config.model.mode == 'stsgcn':
        model = MR_STSGCN(config=config.model)

    print('4. initialize model ...')
    if(config.model.start_epochs>0):
        print('read params:',config.model.init_params)
        model.load_parameters(filename=config.model.init_params, ctx=config.model.ctx)
    else:
        model.initialize(ctx=config.model.ctx, init=MyInit(), force_reinit = True)

    # Optimization
    loss_function = gluon.loss.HuberLoss()
    trainer = gluon.Trainer(model.collect_params(), config.model.optimizer, {'learning_rate':config.model.learning_rate, 'wd':config.model.wd})

    # Traning and testing
    print('5. training ...')
    config.model.logger.open(config.model.log_file,mode="a")
    config.model.logger.write(config.model.name,'\n\n')
    config.model.logger.write(config.model.workname,'\n\n')
    config.model.logger.write('\n%s [START %s] %s\n' % ('-' * 26, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 26))
    config.model.logger.write(f"{'Epoch':^5} | {'TraMAE':^7}{'TraRMSE':^7}{'TraMAPE':^7}| {'ValMAE':^7}{'ValRMSE':^7}{'MinEpoch':^7}| {'TesMAE':^7}{'TesRMSE':^7}{'MinEpoch':^7}|{'TimeCost':^10}\n")
    config.model.logger.write('-' * 80 +'\n')
    
    metrics_tra = Metric(num_prediction=config.model.num_prediction)
    metrics_val = Metric(num_prediction=config.model.num_prediction)
    metrics_tes = Metric(num_prediction=config.model.num_prediction)
    for epoch in range(config.model.start_epochs, config.model.end_epochs):
                
        # training
        train(model, config.data.train_loader, trainer, loss_function, epoch, metrics_tra, config=config.model)

        # validation
        evals(model, config.data.val_loader, epoch, metrics_val, config=config.model, mode='Val')            
        
        if metrics_val.best_metrics['epoch'] == epoch:
            if epoch>80:
                params_filename = config.PATH_LOG+'model/'+f"BestVal-{config.rid}_{config.model.name}.params"
                model.save_parameters(params_filename)
                config.model.logger.write(f'BestVal params -> Epoch:{epoch+1}\n',is_terminal=False)
            # testing
            evals(model, config.data.test_loader, epoch, metrics_tes, config=config.model, mode='Tes')
        else:
            print()
            
        if epoch-metrics_val.best_metrics['epoch']>100:
            break

    config.model.logger.write('\n'+('-'*20)+'\n',is_terminal=True)
    config.model.logger.write(str_config(config)+'\n',is_terminal=True)
    config.model.logger.write(('-'*20)+'\n',is_terminal=True)
    
    # 最终结果记录
    best_val_epoch = metrics_val.best_metrics['epoch']
    config.model.logger.write(f"Best val | Epoch:{best_val_epoch+1} MAE:{metrics_val.best_metrics['mae']:.2f} RMSE:{metrics_val.best_metrics['rmse']:.2f}\n",is_terminal=True)
    config.model.logger.write(f"Best tes | Epoch:{best_val_epoch+1} MAE:{metrics_tes.step_metrics_epoch['mae'][best_val_epoch][-1]:.2f} RMSE:{metrics_tes.step_metrics_epoch['rmse'][best_val_epoch][-1]:.2f}\n",is_terminal=True)
    config.model.logger.write(('-'*20)+'\n',is_terminal=True)
    config.model.logger.write("Corresponding Test:\n",is_terminal=True)
    
    message_lst = metrics_tes.log_lst(epoch=best_val_epoch,sep=',')
    
    for i,m in enumerate(message_lst):
        if i%6==0: print(('-'*20))
        config.model.logger.write(m+'\n',is_terminal=True)

    with open(config.fsummary,mode='a') as f:
        for m in message_lst:
            f.write(f"{config.rid},{m}\n")
        f.close()

    return metrics_val,metrics_tes

def gen_edge_idx_2d(adj, dist):
    E = []
    e_idx_arr = np.zeros(adj.shape)
    idx = np.where(adj > 0)
    e_idx_arr[idx] = np.arange(len(idx[0])) + 1
    for i_idx in range(adj.shape[0]):
        for j_idx in range(adj.shape[1]):
            if adj[i_idx, j_idx] > 0:
                E.append([i_idx, j_idx, dist[i_idx, j_idx]])
    return e_idx_arr, np.array(E)

def default_config(data='Metro', workname='mrstgcn'):

    
    config = edict()
    config.PATH_LOG  = 'result/'
    config.fsummary  = f'{config.PATH_LOG}{workname}_summary.csv'
    config.rid    = 0
    
    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = 'dataset/'
    config.data.feature_file = config.data.path+config.data.name+'/flow.npy'
    
    spatial_edge = 'adj_speed.npy'
    traffic_edge = 'adj_traffic.npy'
    
    config.data.num_graph = 0
    config.data.gfile_lst = []
    
    # I-I and O-O  shape:T,V,V
    config.data.num_graph += 2
    config.data.gfile_lst.append(config.data.path+config.data.name+'/'+spatial_edge)

    # I-O   shape:T,V,V
    config.data.num_graph += 1
    config.data.gfile_lst.append(config.data.path+config.data.name+'/'+traffic_edge)
        
    config.data.num_recent  = 1

    # Metro tra:23day val:3day tes:3day
    if config.data.name=='Metro': 
        config.data.num_features    = 2
        config.data.num_vertices    = 80
        config.data.points_per_hour = 6
        config.data.val_start_idx   = 20*24*6
        config.data.test_start_idx  = (20+3)*24*6

    # chengdu tra:47day val:7day tes:7day
    if config.data.name=='chengdu': 
        config.data.num_features    = 2
        config.data.num_vertices    = 144 # 12x12
        config.data.points_per_hour = 4
        config.data.val_start_idx   = 47*24*4
        config.data.test_start_idx  = (47+7)*24*4

    # Model Config
    config.model = edict()
    config.model.logger     = Logger()
    config.model.ctx_str    = "gpu-0"
    config.model.optimizer  = "adam"
    config.model.learning_rate = 0.001
    config.model.wd         = 1e-5

    config.model.start_epochs  = 0
    config.model.end_epochs    = 200
    config.model.batch_size    = 32
    config.model.init_params   = None
    config.model.ctx = gpu(int(config.model.ctx_str.split('-')[1])) if config.model.ctx_str.startswith('gpu') else cpu()
    
    config.model.huber = 1
    
    config.model.num_stack = 3
    config.model.num_prediction  = 12
    

    config.model.D = 64
    config.model.G = config.data.num_graph
    config.model.V = config.data.num_vertices
    config.model.T = config.model.num_prediction*config.data.num_recent
    
    config.model.dim_edge = 4
    config.model.do_att   = True
    config.model.att_gather_dim = 32
    config.model.dropout    = 0.2
    
    config.model.dist = (1/(1+np.load(config.data.path+config.data.name+'/adj_distance.npy'))).astype(nd_type)
    config.model.gcn_spatial_adj  = np.load(config.data.path+config.data.name+'/adj_spatial.npy').astype(nd_type)
    
    config.model.spatial_edge_idx, config.model.spatial_edge = gen_edge_idx_2d(config.model.gcn_spatial_adj,config.model.dist)
    
    
    if not os.path.exists(config.PATH_LOG+'log/'):
        os.makedirs(config.PATH_LOG+'log/')

    if not os.path.exists(config.PATH_LOG+'model/'):
        os.makedirs(config.PATH_LOG+'model/')

    return config

def str_config(config):
    str_cfg = (f"Data:{config.data.name} Recent:{config.data.num_recent} History:{config.model.T} Predict:{config.model.num_prediction} Batch:{config.model.batch_size} lr:{config.model.learning_rate}\n"
               f"Stack:{config.model.num_stack} Dim:{config.model.D} EdgeDim:{config.model.dim_edge} Att:{config.model.do_att} AttGatherDim:{config.model.att_gather_dim} Drop:{config.model.dropout}"
               )
    return str_cfg

#################################################################

def run_one(args):
    
    workname =f"mr{args.mode}-s{args.stack}-ed{args.ed}-att{args.att}-ad{args.ad}"
    # Metro chengdu
    config = default_config(data=args.data, workname= workname)
    
    config.rid = args.rid
    
    config.model.workname =workname
    
    config.model.wd            = 1e-5
    config.model.batch_size    = 32
    config.model.learning_rate = 0.001
    config.model.end_epochs    = 200

    config.model.mode = args.mode
    config.model.num_stack = args.stack

    config.model.dim_edge = args.ed
    config.model.do_att   = args.att
    config.model.att_gather_dim = args.ad
    config.model.dropout    = 0.2

    config.model.name = (f"{config.data.name}"
                         f"_s{config.model.num_stack}_d{config.model.D}_ed{config.model.dim_edge}"
                         f"_att{int(config.model.do_att)}_ad{config.model.att_gather_dim}"
                         f"_b{config.model.batch_size}_t{config.model.T}p{config.model.num_prediction}"
                         f"_lr{math.ceil(config.model.learning_rate*1000)}e-3"
                         )
    
    config.fsummary           = f"{config.PATH_LOG}{config.model.name}.csv"
    config.model.log_file     = f"{config.PATH_LOG}log/{config.rid}_{config.model.name}.log"
    config.model.fprediction  = f"{config.PATH_LOG}log/{config.rid}_{config.model.name}.npy"

    #  data pre-processing
    print('\n1. data pre-processing ...')
    clean_data = CleanDataset(config)

    # Dataset
    print('2. data loader ...')
    train_data = TrafficDataset(clean_data, (0,config.data.val_start_idx-config.model.num_prediction+1),  config)
    val_data   = TrafficDataset(clean_data, (config.data.val_start_idx,config.data.test_start_idx-config.model.num_prediction+1),  config)
    test_data  = TrafficDataset(clean_data, (config.data.test_start_idx,-1), config)
    # 注意，win的worker应该给0
    config.data.train_loader = gluon.data.DataLoader(train_data,batch_size=config.model.batch_size,shuffle=True)
    config.data.val_loader   = gluon.data.DataLoader(val_data,  batch_size=config.model.batch_size,shuffle=False)
    config.data.test_loader  = gluon.data.DataLoader(test_data, batch_size=config.model.batch_size,shuffle=False)

    main(config)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rid", type=int, required=True)
    parser.add_argument("--mode",   type=str, default='stgcn')
    parser.add_argument("--stack",  type=int, default=3)
    parser.add_argument("--ed",  type=int, default=16)
    parser.add_argument("--ad",  type=int, default=32)
    parser.add_argument("--att", type=int, default=1)
    parser.add_argument("--data",   type=str, default='Metro')
    args = parser.parse_args()
    
    seed = 2019+args.rid-1
    
    np.random.seed(seed)
    random.seed(seed)
    mxrandom.seed(seed)

    
    run_one(args)
