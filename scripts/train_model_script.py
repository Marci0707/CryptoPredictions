import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from _preprocessing import CryptoCompareReader, ColumnLogTransformer, \
    get_not_used_columns, ColumnDropper, DiffTransformer, ManualFeatureEngineer, Pandanizer, \
    PCAFromFirstValidIndex, LinearCoefficientTargetGenerator, ManualValidTargetDetector, WindowsGenerator


def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame, regression_days, scaler, pca,
                    manual_invalidation_pct, window_size, classifier_borders=None):
    # time will not be in training data
    # based on high-low, daily movement will be calculated and the formers will be dropped
    not_used_columns = get_not_used_columns(train_data) + ['time', 'high', 'low']

    column_changer_pipeline = Pipeline(
        [
            ('manual_feature_engineer', ManualFeatureEngineer()),
            ('manual_anomaly_detector', ManualValidTargetDetector(manual_invalidation_pct)),
            ('regression_class_generator', LinearCoefficientTargetGenerator('close', regression_days, 'target',
                                                                            classifier_borders=classifier_borders)),
            ('column_dropper', ColumnDropper(not_used_columns)),
        ]
    )

    train_data = column_changer_pipeline.fit_transform(train_data)
    test_data = column_changer_pipeline.transform(test_data)

    to_scale_columns = train_data.columns.tolist()

    # if the task is classification, dont scale the class indicators
    if classifier_borders is not None:
        to_scale_columns.remove('target')

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
             Pandanizer(columns=to_scale_columns + [col for col in train_data.columns if col not in to_scale_columns])),
            ('pca_social', ColumnTransformer(
                [
                    ('pca', pca, train_data.filter(regex='reddit').columns.tolist())
                ]
                , remainder='passthrough'
            )
             ),
            ('pandanizer2', Pandanizer(
                columns=['social1', 'social2', 'social3'] + [col for col in train_data.columns if
                                                             'reddit' not in col])),
        ]
    )

    train_data = data_transformer_pipeline.fit_transform(train_data)
    test_data = data_transformer_pipeline.transform(test_data)

    window_generator = WindowsGenerator(window_size=window_size,
                                        features=[col for col in train_data.columns if col != 'is_valid_target'],
                                        targets=['close'],
                                        is_valid_target_col_name='is_valid_target')

    train_x, train_y = window_generator.fit_transform(train_data)
    test_x, test_y = window_generator.transform(test_data)

    return train_x, train_y, test_x, test_y


def main():
    pd.set_option('display.expand_frame_repr', False)
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

    train_x, train_y, test_x, test_y = preprocess_data(train_data=train_data, test_data=test_data,
                                                            regression_days=regression_length,
                                                            classifier_borders=classifier_borders,
                                                            scaler=scaler,
                                                            pca=pca,
                                                            manual_invalidation_pct=0.7, window_size=window_size)

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)


if __name__ == '__main__':
    main()
