from typing import Union
import random

import numpy as np
from keras import Sequential, Input
from keras.layers import Dense, Flatten, TimeDistributed, Dropout
from keras.optimizers import Adam
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression

import _preprocessing


class RegressionPredictor:

    def __init__(self, class_borders=None, window_size=14, banned_indices=()):
        self.class_borders = class_borders
        self.window_size = window_size
        self.banned_indices = banned_indices

    def predict(self, series):
        """pandas series of the regression variable"""

        indices = np.array(range(len(series)))
        windows = sliding_window_view(indices, self.window_size + 1)  # last one is the target
        filtered_windows = np.array([window for window in windows if window[-1] not in self.banned_indices])

        lm = LinearRegression(fit_intercept=False)

        def calc_coeffs(x):
            scaled = (x - x[0]) / np.std(x)

            lm.fit(np.array(list(range(len(scaled)))).reshape(-1, 1), scaled)
            return lm.coef_[0]

        regressions = []
        for window in filtered_windows:
            values = series.iloc[window].values
            regression = np.apply_along_axis(calc_coeffs, -1, values)
            regressions.append(regression)

        regressions = np.array(regressions)
        if self.class_borders is not None:
            classes = _preprocessing.discretize(regressions, self.class_borders)
            return classes.reshape(-1, 1)
        else:
            return regressions.reshape(-1, 1)


class RandomBinaryPredictor:

    def __init__(self, predict_feature_idx: int, match_distribution=True):
        self.feature_idx = predict_feature_idx
        self.match_distribution = match_distribution
        self.guess_distribution = np.array([0.5, 0.5])

    def predict(self, X: np.ndarray):
        """X shape: [samples, timesteps, features]"""

        arr = X[:, -1, self.feature_idx].flatten()
        values, counts = np.unique(arr, return_counts=True)

        if self.match_distribution:
            # returns the values as sorted so 0 will be at the first index

            self.guess_distribution = counts / len(arr)

        return random.choices(population=values, weights=self.guess_distribution, k=len(arr))







def create_mlp_baseline(x_train, x_test, n_classes):
    # typically, data_shape ~(samples,17,17) = (samples,window,feature)


    # x_train_reshaped = np.reshape(x_train,(x_train.shape[0],-1))
    # x_test_reshaped = np.reshape(x_test,(x_test.shape[0],-1))
    #


    model = Sequential([
        Input(shape=(x_train.shape[1],x_train.shape[2])),
        Flatten(),
        Dense(units=150, activation='relu',kernel_initializer='HeNormal',kernel_regularizer='l1'),
        Dense(units=50, activation='relu',kernel_initializer='HeNormal',kernel_regularizer='l1'),
        Dense(units=25, activation='relu',kernel_initializer='HeNormal',kernel_regularizer='l1'),
        Dropout(0.2),
        Dense(units=n_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model, x_train, x_test
