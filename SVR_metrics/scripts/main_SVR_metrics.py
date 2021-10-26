#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:06:41 2021

@author: edgar
"""

from dataset_SVR_metrics_ELA import dataset_SVR_metrics_ELA
from get_optim_metrics_ELA import get_optim_metrics_ELA

'''
Esta seccion llama a la función que efectua el GridSearchCV y nos develve el un dataset de los mejores parametros
y las mejores metricas de cada iteracion.
Hay que indicar el nombre del dataset a leer y el nombre con el que sera guardado el dataset de resultados
'''

'''
dataset_SVR_metrics_ELA(dataset_name = 'regression-datasets/mpg.csv', iter = 5,
                        k_folds= 3, save_csv_name = 'metrics_datasets/test_mpg_metrics.csv')
'''






'''
Esta funccion calcula las medias y las varianzas del dataset de las metricas obtenido con la funcion anterior
'''

get_optim_metrics_ELA(file_to_read= 'metrics_datasets/pyrim_metrics.csv', 
                      file_to_create = 'optim_metrics/pyrim_metrics_optim.csv')