#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 18:43:54 2021

@author: edgar
"""
import os; os.system('clear')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge


#Lecturas de datasets originales para obtener las y's
###############################################################################
df_housing = pd.read_csv('regression-datasets/housing.csv')
n_cols = len(df_housing.columns)
x_housing = np.array(df_housing.iloc[:,0:n_cols-1])
y_housing = np.array(df_housing.iloc[:,n_cols-1])

df_mpg = pd.read_csv('regression-datasets/mpg.csv')
n_cols = len(df_mpg.columns)
x_mpg = np.array(df_mpg.iloc[:,0:n_cols-1])
y_mpg = np.array(df_mpg.iloc[:,n_cols-1])

df_pyrim = pd.read_csv('regression-datasets/pyrim.csv')
n_cols = len(df_pyrim.columns)
x_pyrim = np.array(df_pyrim.iloc[:,0:n_cols-1])
y_pyrim = np.array(df_pyrim.iloc[:,n_cols-1])
###############################################################################

#Generar y's con las SVR optimas
###############################################################################
df_housing_svr = pd.read_csv('metrics_datasets/SVR_housing_metrics_Optim.csv')
C_housing = df_housing_svr.iloc[0,0]; epsilon_housing = df_housing_svr.iloc[0,1]; gamma_housing = df_housing_svr.iloc[0,2]
y_housing_svr = SVR(C=C_housing, epsilon=epsilon_housing, gamma=gamma_housing).fit(x_housing, y_housing).predict(x_housing)

df_mpg_svr = pd.read_csv('metrics_datasets/SVR_mpg_metrics_Optim.csv')
C_mpg = df_mpg_svr.iloc[0,0]; epsilon_mpg = df_mpg_svr.iloc[0,1]; gamma_mpg = df_mpg_svr.iloc[0,2]
y_mpg_svr = SVR(C=C_mpg, epsilon=epsilon_mpg, gamma=gamma_mpg).fit(x_mpg, y_mpg).predict(x_mpg)

df_pyrim_svr = pd.read_csv('metrics_datasets/SVR_pyrim_metrics_Optim.csv')
C_pyrim = df_pyrim_svr.iloc[0,0]; epsilon_pyrim = df_pyrim_svr.iloc[0,1]; gamma_pyrim = df_pyrim_svr.iloc[0,2]
y_pyrim_svr = SVR(C=C_pyrim, epsilon=epsilon_pyrim, gamma=gamma_pyrim).fit(x_pyrim, y_pyrim).predict(x_pyrim)
###############################################################################

#Generar y's con modelos Lasso optimos
###############################################################################
df_housing_lasso = pd.read_csv('metrics_datasets/Lasso_housing_metrics_Optim.csv')
alpha_housing = df_housing_lasso.iloc[0,0]; tol_housing = df_housing_lasso.iloc[0,1]
y_housing_lasso = Lasso(alpha=alpha_housing).fit(x_housing, y_housing).predict(x_housing)

df_mpg_lasso = pd.read_csv('metrics_datasets/Lasso_mpg_metrics_Optim.csv')
alpha_mpg = df_mpg_lasso.iloc[0,0]; tol_mpg = df_mpg_lasso.iloc[0,1]
y_mpg_lasso = Lasso(alpha=alpha_mpg).fit(x_mpg, y_mpg).predict(x_mpg)

df_pyrim_lasso = pd.read_csv('metrics_datasets/Lasso_pyrim_metrics_Optim.csv')
alpha_pyrim = df_pyrim_lasso.iloc[0,0]; tol_pyrim = df_pyrim_lasso.iloc[0,1]
y_pyrim_lasso = Lasso(alpha=alpha_pyrim).fit(x_pyrim, y_pyrim).predict(x_pyrim)
###############################################################################


legends = ['Original data', 'SVR', 'Lasso']
df_housing_ = pd.DataFrame(columns = legends, data = np.array([y_housing, y_housing_svr, y_housing_lasso]).T).sort_values(by=legends[0])
df_mpg_ = pd.DataFrame(columns = legends, data = np.array([y_mpg, y_mpg_svr, y_mpg_lasso]).T).sort_values(by=legends[0])
df_pyrim_ = pd.DataFrame(columns = legends, data = np.array([y_pyrim, y_pyrim_svr, y_pyrim_lasso]).T).sort_values(by=legends[0])


#Realizamos los gráficos
###############################################################################
fig = plt.figure(figsize = (20,10))

ax1 = fig.add_subplot(1,3,1)
ax1.set_title('Dataset housing')
ax2 = fig.add_subplot(1,3,2)
ax2.set_title('Dataset mpg')
ax3 = fig.add_subplot(1,3,3)
ax3.set_title('Dataset pyrim')



ax1.plot(np.array(df_housing_[legends[0]]))
ax1.plot(np.array(df_housing_[legends[1]]))
ax1.plot(np.array(df_housing_[legends[2]]))
ax2.plot(np.array(df_mpg_[legends[0]]))
ax2.plot(np.array(df_mpg_[legends[1]]))
ax2.plot(np.array(df_mpg_[legends[2]]))
ax3.plot(np.array(df_pyrim_[legends[0]]))
ax3.plot(np.array(df_pyrim_[legends[1]]))
ax3.plot(np.array(df_pyrim_[legends[2]]))



ax1.legend(legends)
ax1.set(xlabel='register', ylabel='Class')
ax2.legend(legends)
ax2.set(xlabel='register', ylabel='Class')
ax3.legend(legends)
ax3.set(xlabel='register', ylabel='Class')



fig.savefig('regression_comparison.png', dpi = 400)
#fig.savefig('regression_comparison.png')
