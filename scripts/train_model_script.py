import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from _preprocessing import CryptoCompareReader, ColumnLogTransformer, \
    get_not_used_columns, ColumnDropper, DiffTransformer, ManualFeatureEngineer, Pandanizer, \
    PCAFromFirstValidIndex, LinearCoefficientTargetGenerator, ManualValidTargetDetector, WindowsGenerator, \
    inverse_scaler_subset
from models.baselines import RegressionPredictor, create_mlp_baseline


def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame, regression_days, scaler, pca,
                    manual_invalidation_pct, window_size, classifier_borders=None):
    # time will not be in training data
    # based on high-low, daily movement will be calculated and the formers will be dropped
    not_used_columns = get_not_used_columns(train_data) + ['time', 'high', 'low']

    column_changer_pipeline = Pipeline(
        [
            ('manual_feature_engineer', ManualFeatureEngineer()),
            ('manual_anomaly_detector', ManualValidTargetDetector(manual_invalidation_pct)),
            ('regression_class_generator', LinearCoefficientTargetGenerator('close', regression_days, 'slope_target',
                                                                            classifier_borders=classifier_borders)),
            ('column_dropper', ColumnDropper(not_used_columns)),
        ]
    )

    train_data_tr = column_changer_pipeline.fit_transform(train_data)
    test_data_tr = column_changer_pipeline.transform(test_data)

    to_scale_columns = test_data_tr.columns.tolist()

    # if the task is classification, dont scale the class indicators
    if classifier_borders is not None:
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
             Pandanizer(columns=to_scale_columns + [col for col in test_data_tr.columns if col not in to_scale_columns])),
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
    # showhow std scalers and pca 'forget' their attributes, could not debug quickly

    train_data_tr = data_transformer_pipeline.fit_transform(train_data_tr)
    scaler.fit(train_data_tr)
    test_data_tr = data_transformer_pipeline.transform(test_data_tr)



    final_feature_names = [col for col in train_data_tr.columns if col != 'is_valid_target']
    window_generator = WindowsGenerator(window_size=window_size,
                                        features=final_feature_names,
                                        targets=['slope_target'],
                                        is_valid_target_col_name='is_valid_target')

    train_x, train_y = window_generator.fit_transform(train_data_tr)
    test_x, test_y = window_generator.transform(test_data_tr)


    test_y = keras.utils.to_categorical(test_y)
    train_y = keras.utils.to_categorical(train_y)

    return train_x, train_y, test_x, test_y,final_feature_names,scaler, window_generator


def main():
    test_reader = CryptoCompareReader('btc', '../splits/test', drop_na_subset=['close'], add_time_columns=True,
                                      drop_last=True)
    train_reader = CryptoCompareReader('btc', '../splits/train', drop_na_subset=['close'], add_time_columns=True,
                                       drop_last=True)
    test_data = test_reader.read().drop(columns=['Unnamed: 0_x'], errors='ignore')
    train_data = train_reader.read().drop(columns='Unnamed: 0_x', errors='ignore')

    regression_length = 10
    classifier_borders = (-0.2, 0.2)
    window_size = 14
    pca = PCAFromFirstValidIndex(n_components=3)
    scaler = StandardScaler()

    train_x, train_y, test_x, test_y, feature_names ,scaler, window_generator = preprocess_data(train_data=train_data.copy(deep=True), test_data=test_data.copy(deep=True),
                                                       regression_days=regression_length,
                                                       classifier_borders=classifier_borders,
                                                       scaler=scaler,
                                                       pca=pca,
                                                       manual_invalidation_pct=2, window_size=window_size)

    model, x_train, x_test = create_mlp_baseline(train_x, test_x, len(classifier_borders)+1)

    hist = model.fit(x_train,train_y,epochs=80,validation_split=0.1)
    ax = pd.DataFrame(hist.history).filter(regex='loss').plot(figsize=(8,5))
    pd.DataFrame(hist.history).filter(regex='accuracy').plot(secondary_y=True,ax=ax)
    plt.show()

    y_preds = model.predict(x_test)
    y_preds_1d = np.argmax(y_preds,axis=1)
    y_test_1d = np.argmax(test_y,axis=1)

    diffs = np.abs(y_preds_1d - y_test_1d)
    print('total difference',sum(diffs))
    print('average difference',sum(diffs)/len(y_preds))
    print('% accuracy',1-np.count_nonzero(diffs)/len(y_preds))




if __name__ == '__main__':
    main()
