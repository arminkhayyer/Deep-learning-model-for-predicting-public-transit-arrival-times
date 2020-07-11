'''

'''


from cleaning import Clean
from models import *
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Route:
    def __init__(self, filename, look_back, epochs, minibatch, verbose, learning_rate, n_points):
        '''
        :param filename: this shows the file name, the file needs to be located in the data folder
        :param look_back: is the number of lagged variables
        :param epochs:
        :param minibatch: is the batch size
        :param verbose: 0 shows no detail of the model while 2 is full detailed model
        :param learning_rate: shows the learning rate for the optimizer
        :param n_points: shows the number of points to plot
        '''
        self.path = "Data/" + filename
        self.look_back = look_back
        self.epochs = epochs
        self.minibatch = minibatch
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.n_points = n_points
        df = Clean(self.path)
        df, self.xlables = Lag_df(df, look_back)
        self.x_train, self.x_test, self.y_train, self.y_test = preprocess(df)
        self.x_train_LSTM, self.x_test_LSTM, self.x_train_ANN, self.x_test_ANN, self.y_train, self.y_test = preprocess_LSTM_MLP(df, self.look_back)
        self.MLR = MLR(self.x_train, self.y_train, self.x_test, self.y_test)
        self.MLP = MLP(self.x_train, self.y_train, self.x_test, self.y_test, self.look_back, self.epochs,self.minibatch, self.verbose, self.learning_rate)
        self.MLP_LSTM = MLP_LSTM(self.x_train_LSTM, self.x_train_ANN, self.y_train, self.x_test_LSTM, self.x_test_ANN , self.y_test, self.look_back, self.epochs, self.minibatch, self.verbose, self.learning_rate)
    def Run(self):
        '''
        this runs all the three model and stores their performance results
        '''
        self.MLR.Train()
        self.MLR.Predict()
        self.MLP.Train()
        self.MLP.Predict()
        self.MLP_LSTM.Train()
        self.MLP_LSTM.Predict()
    def Print_results(self):
        '''

        This function prints the results of the three model
        '''
        print("_________________________________________")
        print(
            tabulate([['LR', np.round(self.MLR.MSE, 2), np.round(self.MLR.MAPE, 2), np.round(self.MLR.r_squared,2), np.round(self.MLR.CPU_time,2) ], ['MLP', np.round(self.MLP.MSE,2), np.round(self.MLP.MAPE, 2), "", np.round(self.MLP.average_time,2)], ['MLP-LSTM', np.round(self.MLP_LSTM.MSE, 2), np.round(self.MLP_LSTM.MAPE, 2), " ", np.round(self.MLP_LSTM.average_time, 2)]], headers=["Model",'MSE', 'MAPE', "R2", "AVG Time Per Epoch"]))
        print("_________________________________________")

    def Plot(self):
        '''

        This function plots the prediction of each model
        '''
        plt.plot(self.y_test[:self.n_points], label="Actual")
        plt.plot(self.MLR.Pred[:self.n_points], label = "MLR")
        plt.plot(self.MLP.Pred[:self.n_points], label="MLP")
        plt.plot(self.MLP_LSTM.Pred[:self.n_points], label="MLP-LSTM")
        plt.legend(loc= "best")
        plt.show()

if __name__ == '__main__':
    filename = "Example.xlsx"
    look_back = 30
    verbose = 2
    epochs = 200
    minibatch = 512
    learning_rate = 0.01
    N_points_to_plot = 200
    route = Route(filename, look_back, epochs, minibatch, verbose, learning_rate, N_points_to_plot)
    route.Run()
    route.Print_results()
    route.Plot()

