#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:27:47 2019

@author: luisernestocolchado
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# DL libraries
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes

# Data Manipulation
import numpy as np 
import pandas as pd
import random
import math

# Files/OS
import os
import copy

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Benchmarking
import time

# Error Analysis
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
#%%
df = pd.read_csv('data/data.csv')
print(df.head())
train = df.iloc[:365*24*4,]

cols_to_plot = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
i = 1
# plot each column
plt.figure(figsize = (10,12))
for col in cols_to_plot:
    plt.subplot(len(cols_to_plot), 1, i)
    plt.plot(train[col])
    plt.title(col, y=0.5, loc='left')
    i += 1
plt.show()
df = df.drop(columns=['No'])
#%%
## ONE-HOT ENCODE WIND DIRECTION 
temp = pd.get_dummies(df['cbwd'], prefix='cbwd')
df = pd.concat([df, temp], axis = 1)
del df['cbwd'], temp

#%%
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#%%
reframed = series_to_supervised(df, 30, 10)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]], axis=1, inplace=True)
print(reframed.head())

#%%
train_size = 365 * 24 * 4
##TRAIN
df_train = reframed.iloc[:(train_size), :].copy()
##TEST
df_test = reframed.iloc[train_size:, :].copy()
#%%
##Scale
scaler = MinMaxScaler(feature_range=(0, 1))
#%%
dfTrain = scaler.fit_transform(df_train)
dfTest = scaler.fit_transform(df_test)

#%%

#%%
