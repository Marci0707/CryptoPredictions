import datetime
import os

import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from keras_tuner import Objective
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

from _common import TrainingConfig
from _preprocessing import PCAFromFirstValidIndex, CryptoCompareReader
from models.transformers import hyperopt_2towers
from scripts.train_model_script import preprocess_data




def main():
    test_reader = CryptoCompareReader('btc', '../splits/test', drop_na_subset=['close'], add_time_columns=True,
                                      drop_last=True)
    train_reader = CryptoCompareReader('btc', '../splits/train', drop_na_subset=['close'], add_time_columns=True,
                                       drop_last=True)
    test_data = test_reader.read().drop(columns=['Unnamed: 0_x'], errors='ignore')
    train_data = train_reader.read().drop(columns='Unnamed: 0_x', errors='ignore')

    training_id = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    training_config = TrainingConfig(
        training_id=training_id,
        regression_days=5,
        classifier_borders=(-0.2, 0.2),
        manual_invalidation_percentile=2,
        window_size=10,
        optimizer=Adam(learning_rate=0.0001)
    )

    pca = PCAFromFirstValidIndex(n_components=3)
    scaler = StandardScaler()

    x_train, y_train, x_test, y_test, feature_names, scaler, window_generator = preprocess_data(
        train_data=train_data.copy(deep=True), test_data=test_data.copy(deep=True),
        scaler=scaler,
        pca=pca,
        config=training_config)

    x_train_trans = np.transpose(x_train, (0, 2, 1))

    tuner = kt.Hyperband(
        hypermodel=hyperopt_2towers,
        objective=Objective("val_f1_score", direction="max"),
        max_epochs=100,
        factor=3,
        hyperband_iterations=3,
        tune_new_entries=True,
        allow_new_entries=True,
    )

    training_id = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    training_dir = os.path.join('..', 'hyperopt', training_id)
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=17, restore_best_weights=True)
    lr_decay = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.4, min_lr=1e-8)
    tb = TensorBoard(os.path.join(training_dir,'tmp','tb_logs0'))

    callbacks = [early_stopping,lr_decay,tb]


    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    d_class_weights = dict(enumerate(class_weights))

    tuner.search([x_train,x_train_trans], y_train, epochs=100, validation_split=0.2,callbacks=callbacks,shuffle=True, class_weight=d_class_weights)
    best_model = tuner.get_best_models()[0]
    best_model.save(training_dir)



if __name__ == '__main__':
    main()
