#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:14:55 2021

@author: edgar
"""
import pandas as pd


def get_optim_metrics1_ELA(file_to_read, file_to_create):
    df = pd.read_csv(file_to_read)

    cols = ['C_mean', 'epsilon_mean', 'gamma_mean', 'r2_mean', 'r2_mad', 'RMSE_mean', 'RMSE_mad', 'MSE_mean', 'MSE_mad', 'cv', 'time_mean']
    data_ = [[df['C'].mean(),
              df['epsilon'].mean(),
              df['gamma'].mean(),
              df['r2'].mean(),
              df['r2'].mad(),
              df['RMSE'].mean(),
              df['RMSE'].mad(),
              df['MSE'].mean(),
              df['MSE'].mad(),
              df['cv'].mean(),
              df['time'].mean()]]
    df_metrics_optim = pd.DataFrame(data = data_, columns = cols, index = None)
    
    df_metrics_optim.to_csv(file_to_create, index = False)
    
    
def get_optim_metrics2_ELA(file_to_read, file_to_create):
    df = pd.read_csv(file_to_read)

    cols = ['alpha_mean', 'tol_mean', 'r2_mean', 'r2_mad', 'RMSE_mean', 'RMSE_mad', 'MSE_mean', 'MSE_mad', 'cv', 'time_mean']
    data_ = [[df['alpha'].mean(),
              df['tol'].mean(),
              df['r2'].mean(),
              df['r2'].mad(),
              df['RMSE'].mean(),
              df['RMSE'].mad(),
              df['MSE'].mean(),
              df['MSE'].mad(),
              df['cv'].mean(),
              df['time'].mean()]]
    df_metrics_optim = pd.DataFrame(data = data_, columns = cols, index = None)
    
    df_metrics_optim.to_csv(file_to_create, index = False)