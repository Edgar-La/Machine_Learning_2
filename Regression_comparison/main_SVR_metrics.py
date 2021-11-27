#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:06:41 2021

@author: edgar
"""
import os; os.system('clear')
import numpy as np
from best_SVR_estimator_ELA import best_SVR_estimator_ELA
from best_Lasso_estimator_ELA import best_Lasso_estimator_ELA
from best_GBR_estimator_ELA import best_GBR_estimator_ELA
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#Diccionario sobre el cual GridSearch buscara los mejores parametros para la SVR
parameters1_ = {'C': [2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4],
              'gamma': [2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4],
              'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

#Diccionario sobre el cual GridSearch buscara los mejores parametros para la regressión Lasso (o Ridge)
#parameters2_ = {'alpha': [0.000401280703194, 0.001610262027561, 0.0021261123339, 0.001219570460159, 0.003706512910922]}
#parameters2_ = {'alpha': [.25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]}
parameters2_ = {'alpha': np.linspace(.00123, .001235, 30)}

#Diccionario sobre el cual GridSearch buscara los mejores parametros para la GBR
parameters3_ = {'learning_rate': np.linspace(.01, .1, 10),
              'n_estimators': np.arange(100,200, 10)}

#Creamos un modelo SVR
svr = SVR(kernel = 'rbf')

#Creamos un modelo Lasso
lasso = Lasso()

#Creamos un modelo GBR
gbr = GradientBoostingRegressor()
xgb = XGBRegressor()
lgbmr = LGBMRegressor()

'''###############################################################################################################
Esta seccion llama a la función que efectua el GridSearchCV y nos develve el un dataset de los mejores parametros
y las mejores metricas de cada iteracion.
Hay que indicar el nombre del dataset a leer y el nombre con el que sera guardado el dataset de resultados
###############################################################################################################'''

#Seccion para SVR
#best_SVR_estimator_ELA(dataset_name = 'regression-datasets/pyrim.csv', id2save_metrics = 'SVR_pyrim',
#                        estimator_ = svr, iter = 30, k_folds= 10, parameters = parameters1_)



#Seccion para Lasso (o Ridge)
#best_Lasso_estimator_ELA(dataset_name = 'regression-datasets/housing.csv', id2save_metrics = 'Lasso_housing',
#                        estimator_ = lasso, iter = 30, k_folds= 10, parameters = parameters2_)


#Seccion para GBR
best_GBR_estimator_ELA(dataset_name = 'regression-datasets/pyrim.csv', id2save_metrics = 'XGB_pyrim',
                        estimator_ = xgb, iter = 15, k_folds= 5, parameters = parameters3_)

