#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:03:38 2021

@author: edgar
"""
import os; os.system('clear')
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
import time
from get_optim_metrics_ELA import get_optim_metrics2_ELA

def best_Lasso_estimator_ELA(dataset_name, id2save_metrics = 'DatasetMetrics',
                             estimator_ = None, iter = 5, k_folds= 2, parameters = None):
    
    #Guardamos el tiempo en una variable para contar el tiempo que le toma correr
    startTime = time.perf_counter()
    
    #Leemos el dataset y lo convertimos en datos manipulables
    df = pd.read_csv(dataset_name)
    n_cols = len(df.columns)
    x = np.array(df.iloc[:,0:n_cols-1])
    y = np.array(df.iloc[:,n_cols-1])
    
    print(df); print('Dimensions dataset: ', n_cols, '\n\n')
    
    
    
    #Creamos un dataset donde se guardaran las metricas de cada iteracion
    df_metrics = pd.DataFrame(columns = ['alpha', 'tol', 'r2', 'RMSE', 'MSE', 'cv', 'time'])
    
    for i in range(iter):
        startTime_iteration = time.perf_counter()
        folds = KFold(n_splits=k_folds, random_state = None, shuffle=True)
        #Le indicamos a GridSearchCV la maquina, los parametros y los pliegues
        sh = GridSearchCV(estimator = estimator_, param_grid = parameters, cv=folds).fit(x, y)
        #Obtenemos el mejor resultado que el GridSearchCV encontro
        best_svr = sh.best_estimator_
    
    
        #Hacemos el predict y calculamos las metricas correspondientes
        y_pred = best_svr.predict(x)
        r_2 = r2_score(y, y_pred)
        RMSE = mean_squared_error(y, y_pred, squared=False)
        MSE = mean_squared_error(y, y_pred)
    
        #Contamos e tiempo que toma cada iteracion
        endTime_iteration = time.perf_counter()
        
        #Mostramos en la terminal los resultados de cada iteracion
        print(['iter:'+str(i), best_svr, 'r2='+str(r_2), 'RMSE='+str(RMSE), 'MSE='+str(MSE),
               'cv='+str(folds.get_n_splits()), 'time='+str(endTime_iteration-startTime_iteration)])
        print('###############################################################################################')
    
        #Guardamos lo obtenido en un dataset
        df_metrics.loc[i] = [best_svr.alpha, best_svr.tol, r_2, RMSE, MSE, folds.get_n_splits(), endTime_iteration-startTime_iteration]
     
    
    #Guardamos el dataframe de metricas en un archivo .csv
    df_metrics.to_csv('metrics_datasets/'+id2save_metrics+'_metrics.csv')
    
    #Mostramos el tiempo que tardo el programa
    print('\nTotal time: {:6.3f} seconds'.format(time.perf_counter() - startTime))
    
    #Esta funcion calcula las medias y los errores del dataset de metricas obtenido con la linea 62
    get_optim_metrics2_ELA(file_to_read= 'metrics_datasets/'+id2save_metrics+'_metrics.csv', 
                          file_to_create = 'metrics_datasets/'+id2save_metrics+'_metrics_Optim.csv')