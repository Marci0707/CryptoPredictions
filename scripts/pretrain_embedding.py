import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense
from keras.losses import MeanSquaredError, mean_squared_error
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from _common import TrainingConfig
from _preprocessing import CryptoCompareReader, PCAFromFirstValidIndex
from scripts.train_model_script import preprocess_data


def get_autoencoder(x_train,DIM):  # shape (timestamp x window x feature)
    last_timestemps = x_train[:, -1, :]


    x = last_timestemps
    y = last_timestemps.copy()

    model = Sequential([
        Input(shape=x_train.shape[-1]),
        Dense(units=DIM, activation='relu', kernel_initializer='HeNormal', kernel_regularizer='l1_l2',name='embedding'),
        Dense(units=x_train.shape[-1])
    ])

    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

    return model, X_train, X_test, y_train, y_test


def log_results(directory, mse,dim):

    mse = np.mean(float(mse.numpy()[0]))

    try:
        with open(os.path.join(directory, 'res.txt'), 'r') as f:
            results = json.load(f)
            key = str(dim)
            if key not in results:
                results[key] = [mse]
            else:
                results[key].append(mse)
    except FileNotFoundError:
            results = {
                str(dim): [mse],
            }

    with open(os.path.join(directory, 'res.txt'), 'w') as f:
        json.dump(results,f)




def update_embedding_efficiency_plot():

    base_dir = os.path.join('..','pretraining')

    with open(os.path.join(base_dir,'res.txt'),'r') as f:
        results = json.load(f)

    medians = []
    dims = []
    for key,value_list in results.items():
        dims.append(key)
        medians.append(np.min(value_list))

    plt.show()
    plt.scatter(dims,medians)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Minimum MSE on pretrainings')
    plt.savefig(os.path.join(base_dir,'efficiency.png'))



def main(DIM):

    training_config = TrainingConfig(
        training_id=DIM,
        regression_days=5,
        classifier_borders=(-0.2, 0.2),
        manual_invalidation_percentile=0.7,
        window_size=10,
        optimizer=Adam(learning_rate=0.01)
    )

    x_train = np.load(os.path.join('..','splits','train','x_preprocessed.npy'),allow_pickle=True)

    print(x_train.shape)

    model, X_train, X_test, y_train, y_test = get_autoencoder(x_train,DIM)

    model.compile(loss=MeanSquaredError()
                  ,optimizer=training_config.optimizer)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    lr_decay = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.1, min_lr=1e-8)


    hist = model.fit(X_train, y_train, epochs=1000, validation_split=0.2, callbacks=[early_stopping, lr_decay])

    preds = model.predict(X_test)

    directory = os.path.join('..','pretraining')

    if not os.path.isdir(directory):
        os.mkdir(directory)

    mse = mean_squared_error(y_true=y_test,y_pred=preds)
    print(float(mse.numpy()[0]))
    if float(mse.numpy()[0]) < 0.01:
        model.save(os.path.join(directory,'best.hdf5'))
        exit()

    # log_results(directory,mse,DIM)



if __name__ == '__main__':

    for i in range(20):
        print('pretraining on embedding dim',8)
        main(8)
    # update_embedding_efficiency_plot()