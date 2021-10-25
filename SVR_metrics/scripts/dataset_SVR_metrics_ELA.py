#Importamos las librerias a utilizar
import os; os.system('clear')
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
import time

def dataset_SVR_metrics_ELA(dataset_name, iter = 5, k_folds= 2, save_csv_name = 'dataset_metrics.csv'):
    
    #Guardamos el tiempo en una variable para contar el tiempo que le toma correr
    startTime = time.perf_counter()
    
    #Leemos el dataset y lo convertimos en datos manipulables
    df = pd.read_csv(dataset_name)
    n_cols = len(df.columns)
    x = np.array(df.iloc[:,0:n_cols-1])
    y = np.array(df.iloc[:,n_cols-1])
    
    print(df); print('Dimensions dataset: ', n_cols, '\n\n')
    #Diccionario sobre el cual GridSearch buscara los mejores parametros
    
    parameters = {'kernel': ['rbf'],
                  'C': [2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4],
                  'gamma': [2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4],
                  'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    
    #Creamos una maquina SVR
    svr = SVR(kernel = 'rbf')
    
    #Creamos un dataset donde se gaudararan las metricas de cada iteracion
    df_metrics = pd.DataFrame(columns = ['Parameters', 'r2', 'RMSE', 'MSE', 'cv', 'time'])
    
    for i in range(iter):
        startTime_iteration = time.perf_counter()
        folds = KFold(n_splits=k_folds, random_state = None, shuffle=True)
        #Le indicamos a GridSearchCV la maquina, los parametros y los pliegues
        sh = GridSearchCV(estimator = svr, param_grid = parameters, cv=folds).fit(x, y)
        #Obtenemos el mejor resultado que el GridSearchCV encontro
        best_svr = sh.best_estimator_
    
    
        #Hacemos el predict y calculamos las metricas correspondientes
        y_pred = best_svr.predict(x)
        r_2 = r2_score(y, y_pred)
        RMSE = mean_squared_error(y, y_pred)
        MSE = mean_squared_error(y, y_pred, squared=False)
    
        #Contamos e tiempo que toma cada iteracion
        endTime_iteration = time.perf_counter()
        
        #Mostramos en la terminall los resultados de cada iteracion
        print(['iter:'+str(i), best_svr, 'r2='+str(r_2), 'RMSE='+str(RMSE), 'MSE='+str(MSE),
               'cv='+str(folds.get_n_splits()), 'time='+str(endTime_iteration-startTime_iteration)])
        print('###############################################################################################')
    
        #Guardamos lo obtenido en un dataset
        df_metrics.loc[i] = [best_svr, r_2, RMSE, MSE, folds.get_n_splits(), endTime_iteration-startTime_iteration]
     
    
    #Guardamos el dataframe de metricas en un archivo .csv
    df_metrics.to_csv(save_csv_name)
    
    #Mostramos el tiempo que tardo el programa
    print('\nTotal time: {:6.3f} seconds'.format(time.perf_counter() - startTime))


