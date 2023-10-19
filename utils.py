# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:21:51 2020

@author: Zhao Yiji
"""

import sys
import numpy as np
from timeit import default_timer as timer

np_type = 'float16'
nd_type = 'float32'


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))



class Metric(object):
    def __init__(self, num_prediction):
        self.time_start = timer()
        self.num_prediction = num_prediction
        self.best_metrics  = {'mae':np.inf, 'rmse':np.inf, 'mape':np.inf, 'loss':np.inf, 'epoch': np.inf}

        self.step_metrics_epoch = {'mae':{},      'rmse':{},      'mape':{}, 
                                   'mae_in':{},   'rmse_in':{},   'mape_in':{},
                                   'mae_out':{},  'rmse_out':{},  'mape_out':{},
                                   'tmae':{},     'trmse':{},     'tmape':{}, 
                                   'tmae_in':{},  'trmse_in':{},  'tmape_in':{},
                                   'tmae_out':{}, 'trmse_out':{}, 'tmape_out':{}
                                   }

    def update_metrics(self, y_true, y_pred):
        # y_true B,T,V,D
        self.metrics = {'mae':0.0, 'rmse':0.0, 'mape':0.0, 'loss':0.0, 'time':0.0}

        n = y_true.shape[0] # num of sample
        true, pred = y_true.reshape(n, -1), y_pred.reshape(n, -1)
        self.metrics['mae'], self.metrics['rmse'], self.metrics['mape'], mse = self.get_metric(true, pred)
        self.metrics['loss'] = mse / 2
        self.metrics['time'] = time_to_str((timer() - self.time_start))

    def update_metrics_in_out(self, y_true, y_pred):
        self.metrics.update({'mae_in':0.0,  'rmse_in':0.0,  'mape_in':0.0,
                             'mae_out':0.0, 'rmse_out':0.0, 'mape_out':0.0})
        n = y_true.shape[0]
        # inflow
        true, pred = y_true[:,:,:,0].reshape(n, -1), y_pred[:,:,:,0].reshape(n, -1)
        self.metrics['mae_in'], self.metrics['rmse_in'], self.metrics['mape_in'], _ = self.get_metric(true, pred)
        # outflow
        true, pred = y_true[:,:,:,1].reshape(n, -1), y_pred[:,:,:,1].reshape(n, -1)
        self.metrics['mae_out'], self.metrics['rmse_out'], self.metrics['mape_out'], _ = self.get_metric(true, pred)


    def update_step_metrics(self, y_true, y_pred, epoch=0):
        idx_lst=['mae','rmse','mape',
                 'mae_in','rmse_in','mape_in','mae_out','rmse_out','mape_out',
                 'tmae','trmse','tmape',
                 'tmae_in','trmse_in','tmape_in','tmae_out','trmse_out','tmape_out']
        
        metrics = {}
        for i in idx_lst:
            metrics[i] = [0.0 for i in range(self.num_prediction)]

        n = y_true.shape[0]
        for t in range(self.num_prediction):
            
            # The total result of t steps. inflow:[0], outflow[1]
            true, pred = y_true[:,:(t+1),:,:].reshape(n, -1), y_pred[:,:(t+1),:,:].reshape(n, -1)
            metrics['mae'][t], metrics['rmse'][t], metrics['mape'][t], _ = self.get_metric(true, pred)
            true, pred = y_true[:,:(t+1),:,0].reshape(n, -1), y_pred[:,:(t+1),:,0].reshape(n, -1)
            metrics['mae_in'][t], metrics['rmse_in'][t], metrics['mape_in'][t], _ = self.get_metric(true, pred)
            true, pred = y_true[:,:(t+1),:,1].reshape(n, -1), y_pred[:,:(t+1),:,1].reshape(n, -1)
            metrics['mae_out'][t], metrics['rmse_out'][t], metrics['mape_out'][t], _ = self.get_metric(true, pred)
    
            # The result of the tth step.
            true, pred = y_true[:,t,:,:].reshape(n, -1), y_pred[:,t,:,:].reshape(n, -1)
            metrics['tmae'][t], metrics['trmse'][t], metrics['tmape'][t], _ = self.get_metric(true, pred)
            true, pred = y_true[:,t,:,0].reshape(n, -1), y_pred[:,t,:,0].reshape(n, -1)
            metrics['tmae_in'][t], metrics['trmse_in'][t], metrics['tmape_in'][t], _ = self.get_metric(true, pred)
            true, pred = y_true[:,t,:,1].reshape(n, -1), y_pred[:,t,:,1].reshape(n, -1)
            metrics['tmae_out'][t], metrics['trmse_out'][t], metrics['tmape_out'][t], _ = self.get_metric(true, pred)
    
        for i in idx_lst:
            self.step_metrics_epoch[i][epoch] = metrics[i]

    def update_best_metrics(self, epoch=0):
        self.best_metrics['mae'],  mae_state  = self.get_best_metric(self.best_metrics['mae'],  self.metrics['mae'])
        self.best_metrics['rmse'], rmse_state = self.get_best_metric(self.best_metrics['rmse'], self.metrics['rmse'])
        self.best_metrics['mape'], mape_state = self.get_best_metric(self.best_metrics['mape'], self.metrics['mape'])
        self.best_metrics['loss'], loss_state = self.get_best_metric(self.best_metrics['loss'], self.metrics['loss'])
 
        if mae_state:
            self.best_metrics['epoch'] = int(epoch)
       
    @staticmethod
    def get_metric(y_true, y_pred):
        mae  = masked_mae_np(y_true, y_pred, 0.0)
        mse  = masked_mse_np(y_true, y_pred, 0.0)
        mape = masked_mape_np(y_true, y_pred, 0.0)
        rmse = mse ** 0.5
        return mae, rmse, mape, mse
        
    @staticmethod
    def get_best_metric(best, candidate, mode='min'):
        state = False
        if mode=='min':
            if candidate < best: 
                best = candidate
                state = True
        else:
            if candidate > best: 
                best = candidate
                state = True
        return best, state
        
    def __str__(self):
        """For print"""
        return f"{self.metrics['mae']:<7.2f}{self.metrics['rmse']:<7.2f}{self.metrics['mape']:<7.2f}{self.metrics['loss']:<10.3f}| {self.best_metrics['mae']:<7.2f}{self.best_metrics['rmse']:<7.2f}{self.best_metrics['epoch']+1:<5}|{self.metrics['time']}"

    def best_str(self):
        """For save"""
        return f"{self.best_metrics['epoch']},{self.best_metrics['mae']:.2f},{self.best_metrics['rmse']:.2f},{self.best_metrics['mape']:.2f}"

    def multi_step_str(self, obj='rmse', sep=',', epoch=0):
        """For print or save""" #"{i+1}:{x:<7.2f}"
        return sep.join([f"{x:.2f}" if sep ==',' else f"{x:<6.2f}" for i,x in enumerate(self.step_metrics_epoch[obj][epoch])])

    def log_lst(self,epoch=None,sep=','):

        message_lst = []

        index = ['mae','mae_in','mae_out','tmae','tmae_in','tmae_out','rmse','rmse_in','rmse_out','trmse','trmse_in','trmse_out','mape','mape_in','mape_out','tmape','tmape_in','tmape_out']
        name  = ['MAEs  ','MAEs-i','MAEs-o','MAEt  ','MAEt-i','MAEt-o','RMSEs  ','RMSEs-i','RMSEs-o','RMSEt  ','RMSEt-i','RMSEt-o','MAPEs  ','MAPEs-i','MAPEs-o','MAPEt  ','MAPEt-i','MAPEt-o']

        for i,n in zip(index,name):
            message_lst.append(f"{n},{self.multi_step_str(obj=i, sep=sep, epoch=epoch)}")

        return message_lst
        
# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=True, is_file=True):
        if '\r' in message: is_file=False

        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file:
            self.file.write(message)
            self.file.flush()

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError