import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from sklearn.metrics import r2_score
import keras
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def Lag_df(dataframe, look_back=1):
    '''
    :param dataframe:
    :param look_back:
    :return: A Dataframe with lagged variables of the duration variable
    '''
    for i in range(1, look_back +1):
        a = str(i)
        dataframe[a] = dataframe['duration'].shift(i)
    dataframe = dataframe.dropna()
    xlables = ['new_on', 'new_off', "LOAD", "DLMILES", "DIR"]
    for i in range(1, look_back + 1):
        xlables.append(str(i))
    return dataframe, xlables


def preprocess(df):
    x_data = (np.array(df.drop(['duration'], 1)))
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_data = scaler.fit_transform(x_data)
    y_data = np.array(df['duration'])
    x_train , x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2,random_state=42,shuffle=False )
    return x_train , x_test, y_train, y_test


def preprocess_LSTM_MLP(df, look_back):
    x_data_ANN = np.array(df[['new_on', 'new_off', "LOAD", "DLMILES", "DIR"]])
    x_data_LSTM = np.array(df.drop(['duration', 'new_on', 'new_off', "LOAD", "DLMILES", "DIR"], 1))
    y_data = np.array(df['duration']).reshape(-1, 1)
    x_data_LSTM = MinMaxScaler(feature_range=(0, 1)).fit_transform(x_data_LSTM)
    x_data_ANN = MinMaxScaler(feature_range=(0, 1)).fit_transform(x_data_ANN)
    x_train_LSTM, x_test_LSTM, y_train, y_test, x_train_ANN, x_test_ANN = train_test_split(x_data_LSTM, y_data,
                                                                                           x_data_ANN, test_size=0.2,
                                                                                           random_state=42,
                                                                                           shuffle=False)
    x_train_LSTM = x_train_LSTM.reshape(-1, 1, look_back)
    x_test_LSTM = x_test_LSTM.reshape(-1, 1, look_back)
    return x_train_LSTM, x_test_LSTM, x_train_ANN, x_test_ANN, y_train, y_test


def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_pred)) * 100
    return mape

def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def R_2(y_true, y_pred):
    return r2_score(y_true, y_pred)


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


