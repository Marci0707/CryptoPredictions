import tensorflow
from keras.layers import Flatten, Dense, Dropout, LSTM, GRU
from tensorflow import keras
from keras import Sequential, Input


class LastValuePredictor:

    def predict(self,X):
        return X[:,-1, 0]

def get_fccn_baseline(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    return model


def get_gru_baseline(input_shape):
    model = Sequential()
    model.add(LSTM(10, input_shape=input_shape, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(30))
    model.add(Dense(1))

    return model

