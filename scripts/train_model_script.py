import datetime
import os
from datetime import time

import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow import keras
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import TrainingConfig
from _preprocessing import CryptoCompareReader, ColumnLogTransformer, \
    get_not_used_columns, ColumnDropper, DiffTransformer, ManualFeatureEngineer, Pandanizer, \
    PCAFromFirstValidIndex, LinearCoefficientTargetGenerator, ManualValidTargetDetector, WindowGenerator, \
    inverse_scaler_subset
from evaluation import viz_history, save_model, eval_results
from models.baselines import RegressionPredictor, create_mlp_baseline, create_lstm_baseline, create_conv_baseline


def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame, scaler: StandardScaler, pca: PCA,
                    config: TrainingConfig):
    # time will not be in training data
    # based on high-low, daily movement will be calculated and the formers will be dropped
    not_used_columns = get_not_used_columns(train_data) + ['time', 'high', 'low']

    column_changer_pipeline = Pipeline(
        [
            ('manual_feature_engineer', ManualFeatureEngineer()),
            ('manual_anomaly_detector', ManualValidTargetDetector(config.manual_invalidation_percentile,config.window_size,config.regression_days)),
            ('regression_class_generator',
             LinearCoefficientTargetGenerator('close', config.regression_days,config.window_size, 'slope_target',
                                              classifier_borders=config.classifier_borders)),
            ('column_dropper', ColumnDropper(not_used_columns)),
        ]
    )

    train_data_tr = column_changer_pipeline.fit_transform(train_data)
    test_data_tr = column_changer_pipeline.transform(test_data)

    to_scale_columns = test_data_tr.columns.tolist()
    to_scale_columns.remove('slope_feature')
    to_scale_columns.remove('slope_target')

    take_log_columns = ['new_addresses', 'active_addresses', 'transaction_count',
                        'average_transaction_value', 'close',
                        'BTCTradedToUSD', 'USDTradedToBTC', 'reddit_active_users',
                        'reddit_comments_per_day']

    data_transformer_pipeline = Pipeline(
        [
            ('log_taker', ColumnLogTransformer(take_log_columns, add_one=True)),
            ('diff_taker',
             DiffTransformer(take_log_columns + ['block_height', 'current_supply'], replace_nan_to_zeros=True)),
            ('scaler', ColumnTransformer(
                [
                    ('scaler', scaler, to_scale_columns)
                ], remainder='passthrough')
             ),
            ('pandanizer1',
             Pandanizer(
                 columns=to_scale_columns + [col for col in test_data_tr.columns if col not in to_scale_columns])),
            ('pca_social', ColumnTransformer(
                [
                    ('pca', pca, test_data_tr.filter(regex='reddit').columns.tolist())
                ]
                , remainder='passthrough'
            )
             ),
            ('pandanizer2', Pandanizer(
                columns=['social1', 'social2', 'social3'] + [col for col in test_data_tr.columns if
                                                             'reddit' not in col])),
        ]
    )
    # somehow std scalers and pca 'forget' their attributes, could not debug quickly

    train_data_tr = data_transformer_pipeline.fit_transform(train_data_tr)
    scaler.fit(train_data_tr)
    test_data_tr = data_transformer_pipeline.transform(test_data_tr)

    final_feature_names = [col for col in train_data_tr.columns if col != 'is_valid_target' and col != 'slope_target']
    window_generator = WindowGenerator(window_size=config.window_size,
                                       features=final_feature_names,
                                       targets=['slope_target'],
                                       is_valid_target_col_name='is_valid_target')

    train_x, train_y = window_generator.fit_transform(train_data_tr)
    window_generator.predicted_indices = []
    test_x, test_y = window_generator.transform(test_data_tr)

    test_y = keras.utils.to_categorical(test_y)
    train_y = keras.utils.to_categorical(train_y)

    return train_x, train_y, test_x, test_y, final_feature_names, scaler,window_generator


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
        optimizer=Adam(learning_rate=0.0005)
    )


    pca = PCAFromFirstValidIndex(n_components=3)
    scaler = StandardScaler()

    x_train, y_train, x_test, y_test, feature_names, scaler,window_generator  = preprocess_data(
        train_data=train_data.copy(deep=True), test_data=test_data.copy(deep=True),
        scaler=scaler,
        pca=pca,
        config=training_config)

    # model = create_lstm_baseline(x_train, len(training_config.classifier_borders) + 1)
    # model = create_mlp_baseline(x_train, len(training_config.classifier_borders) + 1)

    x_train = np.reshape(x_train, (*x_train.shape, 1))
    x_test = np.reshape(x_test, (*x_test.shape, 1))
    model = create_conv_baseline(x_train, len(training_config.classifier_borders) + 1)

    print('x,y shapes',x_train.shape,y_train.shape)
    model.compile(loss='categorical_crossentropy', optimizer=training_config.optimizer, metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=17,restore_best_weights=True)
    lr_decay = ReduceLROnPlateau(monitor='val_accuracy',patience=5,factor=0.4, min_lr=1e-8)


    training_dir = os.path.join('..', 'trainings', training_id)
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)

    hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping,lr_decay], shuffle=True)
    y_preds = model.predict(x_test)

    save_model(model, training_config, training_dir)
    viz_history(hist, training_dir)
    banned_indices = window_generator.banned_indices
    eval_results(y_preds, y_test,test_data, training_dir,banned_indices,training_config.regression_days, class_borders=training_config.classifier_borders)


if __name__ == '__main__':
    main()
