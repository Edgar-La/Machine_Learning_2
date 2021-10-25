#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:06:41 2021

@author: edgar
"""
from dataset_SVR_metrics_ELA import dataset_SVR_metrics_ELA

dataset_SVR_metrics_ELA(dataset_name = 'regression-datasets/pyrim.csv', iter = 10,
                        k_folds= 2, save_csv_name = 'test_pyrim_metrics.csv')