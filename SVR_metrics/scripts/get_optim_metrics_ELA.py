#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:14:55 2021

@author: edgar
"""
import pandas as pd


def get_optim_metrics_ELA(file_to_read, file_to_create):
    df = pd.read_csv(file_to_read)

    cols = ['C_mean', 'epsilon_mean', 'gamma_mean', 'r2_mean', 'r2_var', 'RMSE_mean', 'RMSE_var', 'MSE_mean', 'MSE_var', 'cv', 'time_mean']
    data_ = [[df['C'].mean(),
                               df['epsilon'].mean(),
                               df['gamma'].mean(),
                               df['r2'].mean(),
                               df['r2'].var(),
                               df['RMSE'].mean(),
                               df['RMSE'].var(),
                               df['MSE'].mean(),
                               df['MSE'].var(),
                               df['cv'].mean(),
                               df['time'].mean()]]
    df_metrics_optim = pd.DataFrame(data = data_, columns = cols, index = None)
    
    df_metrics_optim.to_csv(file_to_create, index = False)