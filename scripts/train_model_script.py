import datetime
import os
from datetime import time

import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras, unique_with_counts
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.python.ops.losses.losses_impl import softmax_cross_entropy
from tensorflow_addons.metrics import F1Score

from _common import TrainingConfig
from _preprocessing import CryptoCompareReader, ColumnLogTransformer, \
    get_not_used_columns, ColumnDropper, DiffTransformer, ManualFeatureEngineer, Pandanizer, \
    PCAFromFirstValidIndex, LinearCoefficientTargetGenerator, ManualValidTargetDetector, WindowGenerator, \
    inverse_scaler_subset, decompose_to_features
from evaluation import viz_history, save_model, eval_results
from models.baselines import RegressionPredictor, create_mlp_baseline, create_lstm_baseline, create_conv_baseline
from models.transformers import create_2towers, create_stacked_encoder, create_encoder_block


def balanced_accuracy(y_true,y_pred):
    return balanced_accuracy_score(tf.argmax(y_true,axis=1),tf.argmax(y_pred,axis=1))




def custom_loss(y_true, y_pred):
    cat_cross_loss = CategoricalCrossentropy()(y_true, y_pred)
    return cat_cross_loss

    # all penalize same class predictions accross batch
    y, idx, count = unique_with_counts(y_pred)
    most_common_count = tf.math.reduce_max(count)
    most_common_relative_freq = tf.math.divide(most_common_count, tf.size(y_pred))
    penalty = tf.constant(0, dtype=tf.float32)
    penality_border = 0.8

    if most_common_relative_freq > penality_border:
        penalty = tf.math.subtract(most_common_relative_freq, penality_border)
        penalty = tf.math.divide(penalty, 1 - penality_border)
        penalty = tf.cast(penalty, tf.float32)

    return cat_cross_loss + penalty


def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame, scaler: StandardScaler, pca: PCA,
                    config: TrainingConfig):
    # time will not be in training data
    # based on high-low, daily movement will be calculated and the formers will be dropped
    not_used_columns = get_not_used_columns(train_data) + ['time', 'high', 'low']

    column_changer_pipeline = Pipeline(
        [
            ('manual_feature_engineer', ManualFeatureEngineer()),
            ('manual_anomaly_detector',
             ManualValidTargetDetector(config.manual_invalidation_percentile, config.window_size,
                                       config.regression_days)),
            ('regression_class_generator',
             LinearCoefficientTargetGenerator('close', config.regression_days, config.window_size, 'slope_target',
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
                        'BTCTradedToUSD', 'USDTradedToBTC', 'reddit_active_users',
                        'reddit_comments_per_day']

    data_transformer_pipeline = Pipeline(
        [
            ('log_taker', ColumnLogTransformer(take_log_columns, add_one=True)),
            ('diff_taker',
             DiffTransformer(take_log_columns + ['close', 'block_height', 'current_supply'],
                             replace_nan_to_zeros=True)),
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

    return train_x, train_y, test_x, test_y, final_feature_names, scaler, window_generator


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
        manual_invalidation_percentile=0.7,
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

    np.save('../splits/train/x_preprocessed.npy', x_train)
    np.save('../splits/train/y_preprocessed.npy', y_train)
    np.save('../splits/test/x_preprocessed.npy', x_test)
    np.save('../splits/test/y_preprocessed.npy', y_test)
    print('x,y shapes', x_train.shape, y_train.shape)

    inputs_train = [x_train]
    inputs_test = [x_test]

    ##-------choose a model--------#
    # LSTM#
    model = create_lstm_baseline(x_train, len(training_config.classifier_borders) + 1)

    # MLP#
    # model = create_mlp_baseline(x_train, len(training_config.classifier_borders) + 1)

    # CONV#
    # x_train = np.reshape(x_train, (*x_train.shape, 1))
    # x_test = np.reshape(x_test, (*x_test.shape, 1))
    # model = create_conv_baseline(x_train, len(training_config.classifier_borders) + 1)

    # 2 TOWERS TRANSFORMER
    # x_train_trans = np.transpose(x_train, (0, 2, 1))
    # x_test_trans = np.transpose(x_test, (0, 2, 1))
    # model = create_2towers(x_train, x_train_trans, len(training_config.classifier_borders) + 1)
    # inputs_train = [x_train, x_train_trans]
    # inputs_test = [x_test, x_test_trans]

    # STACKED ENCODERS
    # model = create_stacked_encoder(x_train,len(training_config.classifier_borders)+1)


    # ENCODER BLOCK
    # inputs_train = x_train.reshape((x_train.shape[0],-1,1))
    # inputs_test = x_test.reshape((x_test.shape[0],-1,1))
    # inputs_train = x_train
    # inputs_test = x_test
    # model = create_encoder_block(inputs_train,len(training_config.classifier_borders)+1)


    ##-------choose a model--------#
    model.compile(loss=softmax_cross_entropy, optimizer=training_config.optimizer, metrics=['accuracy',F1Score(num_classes=len(training_config.classifier_borders)+1,average='macro')])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=17, restore_best_weights=True)
    lr_decay = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.4, min_lr=1e-8)

    training_dir = os.path.join('..', 'trainings', training_id)
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)

    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    d_class_weights = dict(enumerate(class_weights))

    hist = model.fit(inputs_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping, lr_decay],
                     shuffle=True, class_weight=d_class_weights)
    # hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping,lr_decay], shuffle=True)

    y_preds = model.predict(inputs_test)
    # y_preds = model.predict(x_test)

    save_model(model, training_config, training_dir)
    viz_history(hist, training_dir)
    banned_indices = window_generator.banned_indices
    eval_results(y_preds, y_test, test_data, training_dir, banned_indices, training_config.regression_days,
                 class_borders=training_config.classifier_borders)


if __name__ == '__main__':
    main()
