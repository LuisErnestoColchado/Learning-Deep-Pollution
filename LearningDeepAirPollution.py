##
# title: Learning Deep Air Pollution
# description:
# start date: April 26, 2019
# author: Luis Ernesto Colchado Soncco
# email: luis.colchado@ucsp.edu.pe

##
# DL libraries
import tensorflow as tf

#LSTMED
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

# Error Analysis
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

# ARIMA
from statsmodels.tsa.arima_model import ARIMA

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
# DATA WITHOUT MONTH, DAY AND HOUR
data1 = df.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']].values.copy()
# DATA WITH MONTH, DAY AND HOUR
data2 = df.values.copy()

##
X = data1
Y = df.loc[:,'pm2.5'].values.copy().reshape(df.shape[0],1)

## z-score transform x - not including those one-hot columns!
for i in range(0,7):
    currentFeature = X[:, i].reshape(X.shape[0],1)
    X[:, i] = scalar.fit_transform(currentFeature).reshape(X.shape[0])

## z-score transform y
Y = scalar.fit_transform(Y)

##
input_seq_len = 8
output_seq_len = 1

def generate_train_samples(x, y, batch_size=24, input_seq_len=input_seq_len,
                           output_seq_len=output_seq_len,replace=False):
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace=replace)

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
def arimaModel(train,test):
    predictions = []
    history = [x for x in train]
    print("len test",len(test))
    for t in range (0,len(test)):
        print("time: ", t)
        model = ARIMA(history, order=(8, 1, 0))
        modelFit = model.fit(maxiter=10, disp=0)
        output = modelFit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print("RMSE ARIMA: ", rmse)

##
# CROSS VALIDATION - ARIMA
kFolds = KFold(n_splits=5)
countCross = 1
for train, test in kFolds.split(X):
    Train = X[train, 1]
    Test = X[test, 1]
    print("CROSS VALIDATION ", countCross)
    arimaModel(Train, Test)
    countCross+=1

## LSTM NN
# LSTM
learning_rate_lstm = np.power(10.0,-2.0)
training_steps = 100
display_step = 100
num_input = X.shape[1]
timesteps = input_seq_len
num_units = 30
num_classes = 1
num_layers = 2
batch_size = 16
_lambda = 0.003
tf.reset_default_graph()

x = tf.placeholder("float", [None, timesteps, X.shape[1]])
y = tf.placeholder("float", [None, output_seq_len, num_classes])

keep_prob = 0.5

weights = {
    'out': tf.Variable(tf.random_normal([num_units, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

initializer = tf.random_uniform_initializer(-1, 1)


def RNN(x, weights, biases):
    inp = tf.unstack(x ,timesteps,1)
    # track through the layers
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer), reuse=tf.AUTO_REUSE):
            # forward cells
            lstm_layer = tf.contrib.rnn.LSTMCell(num_units,initializer=initializer)
            outputs, _ = rnn.static_rnn(lstm_layer, inp, dtype="float32")

    #rnn_outputs_fw = tf.reshape(output_fw, [-1, num_units])
    #rnn_outputs_bw = tf.reshape(output_bw, [-1, num_units])
    output = tf.matmul(outputs[-1], weights['out']) + biases['out']
    #out_bw = tf.matmul(rnn_outputs_bw, weights['out']) + biases['out']
    return output


logits = RNN(x, weights, biases)

prediction = tf.nn.sigmoid(logits)

# Training loss and optimizer

loss = tf.reduce_mean(tf.pow(prediction - y, 2))

# L2 regularization for weights and biasesreg_loss = 0
regularizers = tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out'])

loss = loss + _lambda * regularizers

train_op = tf.train.AdamOptimizer(learning_rate_lstm).minimize(loss)
# Add the ops to initialize variables.  These will include
# the optimizer slots added by AdamOptimizer().
init_op = tf.initialize_all_variables()

##
# CROSS VALIDATION - LSTM

kFolds = KFold(n_splits=5)
countCross = 1
for train, test in kFolds.split(X):
    xTrain = X[train,:]
    yTrain = Y[train,:]
    xTest = X[test,:]
    yTest = Y[test,:]
    print("CROSS VALIDATION ", countCross)
    # CREATE LIST FOR SAVE TRAINING AND VALIDATION LOSSES
    train_losses = []
    val_losses = []

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        print("Training losses: ")
        for i in range(training_steps):
            batch_input, batch_output = generate_train_samples(xTrain,yTrain,batch_size=batch_size)
            _, loss_t = sess.run([train_op, loss], feed_dict={x: batch_input, y: batch_output})
            print(loss_t)
        #init = tf.global_variables_initializer()
        test_x, test_y = generate_test_samples(xTest, yTest)

        preds = sess.run(prediction, feed_dict={x: test_x, y: test_y})

        inv_test = scalar.inverse_transform(test_y.reshape(test_y.shape[0],1))
        inv_preds = scalar.inverse_transform(preds.reshape(preds.shape[0],1))

        print("Test rmse is: ", np.sqrt(np.mean((inv_preds - inv_test) ** 2)))
##
# LSTM ENCODER DECODER
# LSTM ENCODER: USE A LSTM FOR PROCESSING THE HISTORY DATA
learning_rate = 0.01
lambda_l2_reg = 0.003

# length of input signals
input_seq_len = input_seq_len
# length of output signals
output_seq_len = output_seq_len
# size of LSTM Cell
hidden_dim = 30
# num of input signals
input_dim = X.shape[1]
# num of output signals
output_dim = Y.shape[1]
# num of stacked lstm layers
num_stacked_layers = 2
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5

##
def LSTMED(feed_previous=False):
    tf.reset_default_graph()

    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    weights = {
        'out': tf.get_variable('Weights_out', \
                               shape=[hidden_dim, output_dim], \
                               dtype=tf.float32, \
                               initializer=tf.truncated_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out', \
                               shape=[output_dim], \
                               dtype=tf.float32, \
                               initializer=tf.constant_initializer(0.)),
    }

    with tf.variable_scope('Seq2seq'):
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
            for t in range(input_seq_len)
        ]

        # Decoder: target outputs
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
            for t in range(output_seq_len)
        ]

        # Give a "GO" token to the decoder.
        # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
        # first element will be fed as decoder input which is then 'un-guided'
        dec_inp = [tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO")] + target_seq[:-1]

        with tf.variable_scope('LSTMCell'):
            cells = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        def _rnn_decoder(decoder_inputs,
                         initial_state,
                         cell,
                         loop_function=None,
                         scope=None):
            """RNN decoder for the sequence-to-sequence model.
            Args:
              decoder_inputs: A list of 2D Tensors [batch_size x input_size].
              initial_state: 2D Tensor with shape [batch_size x cell.state_size].
              cell: rnn_cell.RNNCell defining the cell function and size.
              loop_function: If not None, this function will be applied to the i-th output
                in order to generate the i+1-st input, and decoder_inputs will be ignored,
                except for the first element ("GO" symbol). This can be used for decoding,
                but also for training to emulate http://arxiv.org/abs/1506.03099.
                Signature -- loop_function(prev, i) = next
                  * prev is a 2D Tensor of shape [batch_size x output_size],
                  * i is an integer, the step number (when advanced control is needed),
                  * next is a 2D Tensor of shape [batch_size x input_size].
              scope: VariableScovpe for the created subgraph; defaults to "rnn_decoder".
            Returns:
              A tuple of the form (outputs, state), where:
                outputs: A list of the same length as decoder_inputs of 2D Tensors with
                  shape [batch_size x output_size] containing generated outputs.
                state: The state of each cell at the final time-step.
                  It is a 2D Tensor of shape [batch_size x cell.state_size].
                  (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                   states can be the same. They are different for LSTM cells though.)
            """
            with variable_scope.variable_scope(scope or "rnn_decoder"):
                state = initial_state
                outputs = []
                prev = None
                for i, inp in enumerate(decoder_inputs):
                    if loop_function is not None and prev is not None:
                        with variable_scope.variable_scope("loop_function", reuse=True):
                            inp = loop_function(prev, i)
                    if i > 0:
                        variable_scope.get_variable_scope().reuse_variables()
                    output, state = cell(inp, state)
                    outputs.append(output)
                    if loop_function is not None:
                        prev = output
            return outputs, state

        def _basic_rnn_seq2seq(encoder_inputs,
                               decoder_inputs,
                               cell,
                               feed_previous,
                               dtype=dtypes.float32,
                               scope=None):
            """Basic RNN sequence-to-sequence model.
            This model first runs an RNN to encode encoder_inputs into a state vector,
            then runs decoder, initialized with the last encoder state, on decoder_inputs.
            Encoder and decoder use the same RNN cell type, but don't share parameters.
            Args:
              encoder_inputs: A list of 2D Tensors [batch_size x input_size].
              decoder_inputs: A list of 2D Tensors [batch_size x input_size].
              feed_previous: Boolean; if True, only the first of decoder_inputs will be
                used (the "GO" symbol), all other inputs will be generated by the previous
                decoder output using _loop_function below. If False, decoder_inputs are used
                as given (the standard decoder case).
              dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
              scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
            Returns:
              A tuple of the form (outputs, state), where:
                outputs: A list of the same length as decoder_inputs of 2D Tensors with
                  shape [batch_size x output_size] containing the generated outputs.
                state: The state of each decoder cell in the final time-step.
                  It is a 2D Tensor of shape [batch_size x cell.state_size].
            """
            with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                enc_cell = copy.deepcopy(cell)
                _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                if feed_previous:
                    return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                else:
                    return _rnn_decoder(decoder_inputs, enc_state, cell)

        def _loop_function(prev, _):
            '''Naive implementation of loop function for _rnn_decoder. Transform prev from
            dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
            used as decoder input of next time step '''
            return tf.matmul(prev, weights['out']) + biases['out']

        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            enc_inp,
            dec_inp,
            cell,
            feed_previous=feed_previous
        )

        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(
            loss=loss,
            learning_rate=learning_rate,
            global_step=global_step,
            optimizer='Adam',
            clip_gradients=GRADIENT_CLIPPING)

    saver = tf.train.Saver

    return dict(
        enc_inp=enc_inp,
        target_seq=target_seq,
        train_op=optimizer,
        loss=loss,
        saver=saver,
        reshaped_outputs=reshaped_outputs,
    )

##
# CROSS VALIDATION - LSTMED
epochs = 100
# batch_size = 16
batch_size = 16
KEEP_RATE = 0.5
kFolds = KFold(n_splits=5)
countCross = 1
for train, test in kFolds.split(X):
    xTrain = X[train,:]
    yTrain = Y[train,:]
    xTest = X[test,:]
    yTest = Y[test,:]
    print("CROSS VALIDATION ", countCross)
    # CREATE LIST FOR SAVE TRAINING AND VALIDATION LOSSES
    train_losses = []
    val_losses = []

    x = np.linspace(0, 40, 130)
    train_data_x = x[:110]

    rnn_model = LSTMED(feed_previous=False)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        print("Training losses: ")
        for i in range(epochs):
            batch_input, batch_output = generate_train_samples(xTrain,yTrain,batch_size=batch_size)

            feed_dict = {rnn_model['enc_inp'][t]: batch_input[:, t] for t in range(input_seq_len)}
            feed_dict.update({rnn_model['target_seq'][t]: batch_output[:, t] for t in range(output_seq_len)})
            _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
            print(loss_t)

        temp_saver = rnn_model['saver']()
        save_path = temp_saver.save(sess, os.path.join('./', 'multivariate_ts_pollution_case'))

    print("Checkpoint saved at: ", save_path)

    test_x, test_y = generate_test_samples(xTest,yTest)
    rnn_model = LSTMED(feed_previous=True)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        saver = rnn_model['saver']().restore(sess, os.path.join('./', 'multivariate_ts_pollution_case'))

        feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)}  # batch prediction
        feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in
                          range(output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

        final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
        final_preds = np.concatenate(final_preds, axis=1)
        inv_test = scalar.inverse_transform(test_y.reshape(test_y.shape[0],1))
        inv_preds = scalar.inverse_transform(final_preds.reshape(final_preds.shape[0],1))
        print("Test rmse is: ", np.sqrt(np.mean((inv_preds - inv_test) ** 2)))
##
from keras.models import Model, Sequential
from keras.layers import Dense, Input, concatenate
from keras.layers import LSTM
from keras.layers.core import Reshape


def lstm(xtrain, ytrain):
    inputs = Input((timesteps, xtrain.shape[1]))

    # ~inputs = Embedding(output_dim=xtrain.shape[0], input_dim=10000, input_length=100)(main_input)

    lstm1 = LSTM(50, input_shape=(timesteps, xtrain.shape[1]), return_sequences=True)
    lstmPm25 = lstm1(inputs)
    #lstm2 = LSTM(50, input_shape=(timesteps, xtrain.shape[1]), return_sequences=True)(lstmPm25)
    lstmPm25 = Dense(1, activation='sigmoid')(lstmPm25)
    lstmModel1 = Model(inputs, lstmPm25)

    return lstmModel1

##
# CROSS VALIDATION - LSTMED
kFolds = KFold(n_splits=5)
countCross = 1
for train, test in kFolds.split(X):
    xTrain = X[train,:]
    yTrain = Y[train,:]
    xTest = X[test,:]
    yTest = Y[test,:]
    print("------------- CROSS VALIDATION ", countCross, "---------")
    print("**********LTSMNN***********")
    xTrain, yTrain = generate_train_samples(xTrain, yTrain, xTrain.shape[0],replace=True)
    model = Sequential()
    model.add(LSTM(50, input_shape=(xTrain.shape[1], xTrain.shape[2])))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss='mae', optimizer='adam')


    model.fit(xTrain, yTrain, epochs=30, batch_size=1000)

    test_x = generate_test_samples(xTest,yTest)

    yhat = model.predict(xTest)

    yhat = np.reshape(yhat, (xTest.shape[0], xTest.shape[1]))
    predictTest = scalar.inverse_transform(yhat)
    yTestI = scalar.inverse_transform(yTest)
    rmseTest = np.sqrt(mean_squared_error(yTestI, predictTest))
    print("RMSE ", rmseTest)
    countCross += 1


##

