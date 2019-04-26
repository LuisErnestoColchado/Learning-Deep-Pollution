##
# title: Learning Deep Air Pollution
# description:
# start date: April 26, 2019
# author: Luis Ernesto Colchado Soncco
# email: luis.colchado@ucsp.edu.pe

##
# DL libraries
import tensorflow as tf

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

# Error Analysis
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

scalar = MinMaxScaler((0,1))

##
# READ DATA 2010 - 2014

df = pd.read_csv('/Users/luisernestocolchado/Documents/MasterCSTesis/code/data/data.csv')
df = df.drop(columns=['No'])
print(df.head())

# PLOT EACH COLUMNS
cols_to_plot = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
i = 1
plt.figure(figsize = (10,12))
for col in cols_to_plot:
    plt.subplot(len(cols_to_plot), 1, i)
    plt.plot(df[col])
    plt.title(col, y=0.5, loc='left')
    i += 1
plt.show()
##
# PRE PROCESSING DATA

# NAN TO 0
df.fillna(0, inplace=True)

# ONE-HOT DIRECTION OF WIND
temp = pd.get_dummies(df['cbwd'], prefix='cbwd')
df = pd.concat([df, temp], axis = 1)
del df['cbwd'], temp

# ONE-HOT MONTH, DAY AND HOUR

temp = pd.get_dummies(df['month'], prefix='month')
df = pd.concat([df, temp], axis = 1)
del df['month'], temp

temp = pd.get_dummies(df['day'], prefix='day')
df = pd.concat([df, temp], axis = 1)
del df['day'], temp

temp = pd.get_dummies(df['hour'], prefix='hour')
df = pd.concat([df, temp], axis = 1)
del df['hour'], temp


##
print(df.shape)
X = df.values
Y = df.loc[:,'pm2.5'].values.copy().reshape(df.shape[0],1)
##
input_seq_len = 8
output_seq_len = 1

def generate_train_samples(x, y, batch_size=24, input_seq_len=input_seq_len,
                           output_seq_len=output_seq_len):
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace=False)

    input_batch_idxs = [list(range(i, i + input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis=0)

    output_batch_idxs = [list(range(i + input_seq_len, i + input_seq_len + output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis=0)

    return input_seq, output_seq  # in shape: (batch_size, time_steps, feature_dim)


def generate_test_samples(x, y, input_seq_len=input_seq_len, output_seq_len=output_seq_len):
    total_samples = x.shape[0]

    input_batch_idxs = [list(range(i, i + input_seq_len)) for i in
                        range((total_samples - input_seq_len - output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis=0)

    output_batch_idxs = [list(range(i + input_seq_len, i + input_seq_len + output_seq_len)) for i in
                         range((total_samples - input_seq_len - output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis=0)

    return input_seq, output_seq
##
# ARIMA MODEL
# AR: Auto regression: Model that use the observations and lagged observations to prediction
# I: Integrated: The difference between the observations
# MA: Moving Average: Model that use the residual error between a observation from moving the average applied to lagged observations

def arima():
    

##
kFolds = KFold(n_splits=5)
countCross = 1
for train, test in kFolds.split(df):
    xTrain = X[train,:]
    yTrain = Y[train,:]
    xTest = X[test,:]
    yTest = Y[test,:]
    shapeX, shapeY = generate_train_samples(xTrain,yTrain)
    testX, testY = generate_test_samples(xTest,yTest)
    print("CROSS VALIDATION ", countCross)
    print(shapeX.shape, shapeY.shape)
    print(testX.shape, testY.shape)
    countCross+=1
##
