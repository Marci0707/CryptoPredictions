import datetime
import os
from datetime import time
import tensorflow_addons as tfa
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, SquaredHinge
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
from keras.utils import losses_utils
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
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
from models.transformers import create_stacked_encoder, create_encoder_block


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame, scaler: StandardScaler, pca: PCA,
                    config: TrainingConfig, halflife: int):
    # time will not be in training data
    # based on high-low, daily movement will be calculated and the formers will be dropped
    not_used_columns = get_not_used_columns(train_data) + ['time', 'high', 'low']

    column_changer_pipeline = Pipeline(
        [
            ('manual_feature_engineer', ManualFeatureEngineer(halflife)),
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

    columns_to_drop = ['Unnamed: 0_y', 'Unnamed: 0', 'block_height', 'current_supply']
    final_feature_names = [col for col in final_feature_names if col not in columns_to_drop]

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


def main(training_group_id: str, model_name: str, halflife: int, tp: int):
    # test_reader = CryptoCompareReader('btc', '../splits/test', drop_na_subset=['close'], add_time_columns=True,
    #                                   drop_last=True)
    # train_reader = CryptoCompareReader('btc', '../splits/train', drop_na_subset=['close'], add_time_columns=True,
    #                                    drop_last=True)
    # test_data = test_reader.read().drop(columns=['Unnamed: 0_x'], errors='ignore')
    # train_data = train_reader.read().drop(columns='Unnamed: 0_x', errors='ignore')

    training_id = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S') + '_' + training_group_id

    training_config = TrainingConfig(
        training_id=training_id,
        regression_days=5,
        classifier_borders=(0,),
        manual_invalidation_percentile=tp,
        window_size=11,
        optimizer=Adam(learning_rate=0.001)
    )

    # pca = PCAFromFirstValidIndex(n_components=3)
    # scaler = StandardScaler()
    #
    # x_train, y_train, x_test, y_test, feature_names, scaler, window_generator = preprocess_data(
    #     train_data=train_data.copy(deep=True), test_data=test_data.copy(deep=True),
    #     scaler=scaler,
    #     pca=pca,
    #     config=training_config,halflife=halflife)
    #
    # banned_indices_arr = np.array(window_generator.banned_indices)
    # np.save(f'../splits/train/banned_indices_h{halflife}_p{tp}.npy', banned_indices_arr)
    # # balance training set
    # y_dense = np.argmax(y_train, axis=1)
    # decrease_y = np.argwhere(y_dense == 0)
    # increase_y = np.argwhere(y_dense == 1)
    # diff = len(increase_y) - len(decrease_y)
    # filtered_increase = increase_y[diff:]
    # indices = sorted(np.concatenate([decrease_y, filtered_increase]).flatten())
    # x_train = x_train[indices]
    # y_train = y_train[indices]
    #
    #
    # np.save(f'../splits/train/x_preprocessed_h{halflife}_p{tp}.npy', x_train)
    # np.save(f'../splits/train/y_preprocessed_h{halflife}_p{tp}.npy', y_train)
    # np.save(f'../splits/test/x_preprocessed_h{halflife}_p{tp}.npy', x_test)
    # np.save(f'../splits/test/y_preprocessed_h{halflife}_p{tp}.npy', y_test)

    x_train = np.load(f'../splits/train/x_preprocessed_h{halflife}_p{tp}.npy')
    y_train = np.load(f'../splits/train/y_preprocessed_h{halflife}_p{tp}.npy')
    x_test = np.load(f'../splits/test/x_preprocessed_h{0}_p{0}.npy')
    y_test = np.load(f'../splits/test/y_preprocessed_h{0}_p{0}.npy')

    print('x,y shapes train', x_train.shape, y_train.shape)
    print('x,y shapes test', x_test.shape, y_test.shape)

    group_folder = os.path.join('..', 'trainings', training_group_id)
    training_dir = os.path.join(group_folder, training_id)

    if not os.path.isdir(group_folder):
        os.mkdir(group_folder)
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)

    inputs_train = x_train
    inputs_test = x_test

    pretrained_autoencoder = tf.keras.models.load_model('../pretraining/best.hdf5')
    pretrained_embedding = pretrained_autoencoder.get_layer('embedding')

    ##-------choose a model--------#
    # LSTM#
    if model_name == 'lstm':
        model = create_lstm_baseline(x_train, len(training_config.classifier_borders) + 1)

    # MLP#
    elif model_name == 'mlp':
        model = create_mlp_baseline(x_train, len(training_config.classifier_borders) + 1)
        model.get_layer(name='embedding').set_weights(pretrained_embedding.weights)
    # CNN#
    elif model_name == 'cnn':
        x_train = np.reshape(x_train, (*x_train.shape, 1))
        x_test = np.reshape(x_test, (*x_test.shape, 1))
        model = create_conv_baseline(x_train, len(training_config.classifier_borders) + 1)
        inputs_train = x_train
        inputs_test = x_test


    # STACKED ENCODERS
    elif model_name == 'encoder_stack':
        model = create_stacked_encoder(x_train, len(training_config.classifier_borders) + 1)
        model.get_layer(name='embedding').set_weights(pretrained_embedding.weights)

    # ENCODER BLOCK
    elif model_name == 'encoder_block':
        model = create_encoder_block(x_train, len(training_config.classifier_borders) + 1)
        model.get_layer(name='embedding').set_weights(pretrained_embedding.weights)

    else:
        raise ValueError('Invalid model name: ' + model_name)

    ##-------end choose a model--------#
    model.compile(loss=SquaredHinge()
                  , optimizer=training_config.optimizer,
                  metrics=['accuracy',
                           F1Score(num_classes=len(training_config.classifier_borders) + 1, average='weighted')])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
    lr_decay = ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.1, min_lr=1e-8)

    inputs_train, inputs_valid, y_train, y_valid = train_test_split(inputs_train, y_train, test_size=0.2,
                                                                    random_state=42)

    hist = model.fit(inputs_train, y_train, epochs=100, validation_data=(inputs_valid, y_valid),
                     callbacks=[early_stopping, lr_decay],
                     shuffle=True)
    # hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping,lr_decay], shuffle=True)

    y_preds = model.predict(inputs_test)
    # y_preds = model.predict(x_test)

    save_model(model, training_config, training_dir)
    viz_history(hist, training_dir)
    np.save(os.path.join(training_dir, 'pred_h0.npy'),y_preds , allow_pickle=True)

    # banned_indices = np.load(f'../splits/train/banned_indices_h{halflife}_p{tp}.npy')
    # eval_results(y_preds, y_test, test_data, training_dir, banned_indices, training_config.regression_days,
    #              class_borders=training_config.classifier_borders)


if __name__ == '__main__':
    n_runs = 15
    for halflife in (0,):
        for top_percent in (0, 1, 3, 5, 8, 13):
            # for model_name in ('mlp',):
            for model_name in ('mlp', 'cnn', 'lstm', 'encoder_stack', 'encoder_block'):
            # for model_name in ('mlp',):
                group = f'{model_name}_h{halflife}_p{top_percent}_test'

                if group not in os.listdir('../trainings'):
                    print('training',group)
                    if 'lstm' in group:
                        n_runs=10
                    else:
                        n_runs=10
                    for run in range(n_runs):
                        print('----now training', group, f' ({run + 1}/{n_runs})', '----')
                        main(group, model_name, halflife, top_percent)
                        tf.keras.backend.clear_session()
                else:
                    print('skipped',group)
