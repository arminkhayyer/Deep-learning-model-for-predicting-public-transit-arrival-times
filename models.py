import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.models import Sequential, Model
from keras.layers import LSTM, Flatten, Input, Embedding, concatenate, Concatenate, average, maximum, Dropout, Dense
from keras import optimizers
from utils import *



class PredictiveModel(object):
    def __init__(self, x_train,y_train,x_test, y_test):
        '''
        :param x_train:  training set
        :param y_train: ground truth labels
        :param x_test: testing set
        :param y_test: ground truth lables for test set
        '''
        self.MSE = 0
        self.MAPE = 0
        self.name = 0
        self.r_squared = 0
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.Pred = []
        self.time_callback = TimeHistory()


class MLR(PredictiveModel):

    def __init__(self,  x_train,y_train,x_test, y_test):
        PredictiveModel.__init__(self, x_train,y_train,x_test, y_test)
        self.name = "Linear Regression"
        self.model = LinearRegression()

    def Train(self):
        start = time.time()
        self.model.fit(self.x_train, self.y_train)
        end = time.time()
        self.CPU_time = end - start

    def Predict(self):
        self.Pred = self.model.predict(self.x_test)
        self.MSE = MSE(y_true=self.y_test, y_pred= self.Pred)
        self.MAPE = MAPE(y_true=self.y_test, y_pred=self.Pred)
        self.r_squared = R_2(self.y_test, self.Pred)

class MLP(PredictiveModel):
    def __init__(self, x_train, y_train, x_test, y_test, look_back, epochs, minibatch, verbose, learning_rate):
        PredictiveModel.__init__(self, x_train, y_train, x_test, y_test)
        self.name = "Multilayer Precetron "
        self.look_back = look_back
        self.epochs = epochs
        self.minibatch = minibatch
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=5 + self.look_back, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1))
        self.optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.MeanSquaredError())
    def Train(self):
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.epochs, verbose=self.verbose, callbacks=[self.time_callback])
        self.average_time = np.mean(self.time_callback.times)
    def Predict(self):
        # make predictions
        self.Pred = self.model.predict(self.x_test)
        self.train_pred = self.model.predict(self.x_train).flatten()
        self.MSE = MSE(y_true=self.y_test, y_pred=self.Pred)
        self.MAPE = MAPE(y_true=self.y_test, y_pred=self.Pred)

class MLP_LSTM(PredictiveModel):
    def __init__(self, x_train_LSTM, x_train_MLP, y_train, x_test_LSTM, x_test_MLP , y_test, look_back, epochs, minibatch, verbose, learning_rate):
        PredictiveModel.__init__(self, x_train_LSTM, y_train, x_test_LSTM, y_test)
        self.name = "Multilayer Precetron "
        self.look_back = look_back
        self.epochs = epochs
        self.minibatch = minibatch
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.MLP_inputs = Input(shape=(5,))
        self.x_train_LSTM = x_train_LSTM
        self.x_train_MLP = x_train_MLP
        self.x_test_LSTM = x_test_LSTM
        self.x_test_MLP = x_test_MLP
        x = Dense(10, activation='relu')(self.MLP_inputs )
        x = Dense(10, activation='relu')(x)
        self.MLP_output = Dense(10, name='ANN_output', activation="relu")(x)

        self.LSTM_input = Input(shape=(1, look_back))
        l = LSTM(20, return_sequences=True)(self.LSTM_input)
        l = LSTM(20, return_sequences=True)(l)
        self.LSTM_output = Flatten()(l)
        main_output = Dense(10, name="main_output", activation="relu")(concatenate([self.MLP_output, self.LSTM_output]))
        self.main_output = Dense(1)(main_output)

        self.optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model = Model(inputs=[self.MLP_inputs, self.LSTM_input], outputs=self.main_output)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

    def Train(self):
        self.model.fit([self.x_train_MLP, self.x_train_LSTM], self.y_train, epochs=self.epochs, batch_size=self.minibatch,
                       verbose=self.verbose , callbacks=[self.time_callback])
        self.average_time = np.mean(self.time_callback.times)
    def Predict(self):
        # make predictions
        self.train_pred = self.model.predict([self.x_train_MLP, self.x_train_LSTM])
        self.Pred = self.model.predict([self.x_test_MLP, self.x_test_LSTM])
        self.MSE = MSE(y_true=self.y_test, y_pred=self.Pred)
        self.MAPE = MAPE(y_true=self.y_test, y_pred=self.Pred)
