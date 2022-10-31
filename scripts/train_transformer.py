import datetime
import os
from datetime import time
import tensorflow_addons as tfa
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Add
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
from keras.utils import losses_utils
from keras_nlp.layers import SinePositionEncoding
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
    inverse_scaler_subset, decompose_to_features, WindowGeneratorRegression
from evaluation import viz_history, save_model, eval_results
from models.baselines import RegressionPredictor, create_mlp_baseline, create_lstm_baseline, create_conv_baseline
from models.transformer_impl import Transformer


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

    columns_to_drop = ['Unnamed: 0_y', 'Unnamed: 0', 'block_height', 'current_supply']
    final_feature_names = [col for col in final_feature_names if col not in columns_to_drop]

    window_generator = WindowGeneratorRegression(window_size=config.window_size,
                                                 features=final_feature_names,
                                                 targets=['close'],
                                                 is_valid_target_col_name='is_valid_target',
                                                 regression_ahead=config.regression_days)

    train_x, train_y = window_generator.fit_transform(train_data_tr)
    window_generator.predicted_indices = []
    test_x, test_y = window_generator.transform(test_data_tr)

    return train_x, train_y, test_x, test_y, final_feature_names, scaler, window_generator


def main():
    test_reader = CryptoCompareReader('btc', '../splits/test', drop_na_subset=['close'], add_time_columns=True,
                                      drop_last=True)
    train_reader = CryptoCompareReader('btc', '../splits/train', drop_na_subset=['close'], add_time_columns=True,
                                       drop_last=True)
    test_data = test_reader.read().drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], errors='ignore')
    train_data = train_reader.read().drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], errors='ignore')

    training_id = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S') + '_transformer'

    training_config = TrainingConfig(
        training_id=training_id,
        regression_days=5,
        classifier_borders=(0,),
        manual_invalidation_percentile=0.5,
        window_size=11,
        optimizer=Adam(learning_rate=0.001)
    )

    pca = PCAFromFirstValidIndex(n_components=3)
    scaler = StandardScaler()

    x_train, y_train, x_test, y_test, feature_names, scaler, window_generator = preprocess_data(
        train_data=train_data.copy(deep=True), test_data=test_data.copy(deep=True),
        scaler=scaler,
        pca=pca,
        config=training_config)

    y_dense = np.argmax(y_test, axis=1)
    decrease_y = np.argwhere(y_dense == 0)
    increase_y = np.argwhere(y_dense == 1)

    diff = len(increase_y) - len(decrease_y)
    filtered_increase = increase_y[diff:]
    indices = sorted(np.concatenate([decrease_y, filtered_increase]).flatten())

    x_train = x_train[indices]
    y_train = y_train[indices]

    np.save('../splits/train/x_preprocessed_transformer.npy', x_train)
    np.save('../splits/train/y_preprocessed_transformer.npy', y_train)
    np.save('../splits/test/x_preprocessed_transformer.npy', x_test)
    np.save('../splits/test/y_preprocessed_transformer.npy', y_test)
    print('x,y shapes', x_train.shape, y_train.shape)
    print(feature_names)

    group_folder = os.path.join('..', 'trainings', 'transformer')
    training_dir = os.path.join(group_folder, training_id)

    if not os.path.isdir(group_folder):
        os.mkdir(group_folder)
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)

    pretrained_autoencoder = tf.keras.models.load_model('../pretraining/best.hdf5')
    pretrained_embedding = pretrained_autoencoder.get_layer('embedding')

    x_input = Input(x_train.shape[1:])
    y_input = Input(y_train.shape[1:])

    pretrained_embedding_x = pretrained_embedding(x_input)
    PE_x = SinePositionEncoding()(pretrained_embedding_x)
    PE_x = Add()([PE_x, pretrained_embedding_x])

    PE_y = SinePositionEncoding()(y_input)
    PE_y = Add()([PE_y, y_input])

    transformer = Transformer(num_layers=2, num_heads=1, dff=12, dropout_rate=0.1, d_model=8, target_vocab_size=1,
                              input_vocab_size=8)([PE_x, PE_y])

    model = Model([x_input, y_input], transformer)
    print(model.summary())


if __name__ == '__main__':
    main()
