import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from _preprocessing import CryptoCompareReader, ColumnLogTransformer, \
    get_not_used_columns, ColumnDropper, DiffTransformer, ManualFeatureEngineer, Pandanizer, \
    PCAFromFirstValidIndex


def main():
    reader = CryptoCompareReader('btc', '../cryptoCompareData', drop_na_subset=['close'], add_time_columns=True,
                                 drop_last=True)
    df = reader.read()

    # time will not be in training data
    # based on high-low, daily movement will be calculated and the formers will be dropped
    not_used_columns = get_not_used_columns(df) + ['time','high','low']

    column_changer_pipeline = Pipeline(
        [
            ('manual_feature_engineer', ManualFeatureEngineer()),
            ('column_dropper', ColumnDropper(not_used_columns))
        ]
    )
    df = column_changer_pipeline.fit_transform(df)
    to_scale_columns = df.columns.tolist()
    take_log_columns = ['new_addresses', 'active_addresses', 'transaction_count',
                        'average_transaction_value', 'close',
                        'BTCTradedToUSD', 'USDTradedToBTC','reddit_active_users',
                        'reddit_comments_per_day']

    data_transformer_pipeline = Pipeline(
        [
            ('log_taker', ColumnLogTransformer(take_log_columns, add_one=True)),
            ('diff_taker', DiffTransformer(take_log_columns + ['block_height', 'current_supply'],replace_nan_to_zeros=True)),
            ('scaler', ColumnTransformer(
                [
                    ('scaler',StandardScaler(),to_scale_columns)
                ],remainder='passthrough')
            ),
            ('pandanizer1', Pandanizer(columns=to_scale_columns + [col for col in df.columns if col not in to_scale_columns])),
             ('pca_social', ColumnTransformer(
                 [
                     ('pca',PCAFromFirstValidIndex(n_components=3),df.filter(regex='reddit').columns.tolist())
                 ]
              ,remainder='passthrough'
             )
              ),
            ('pandanizer2',Pandanizer(columns=['social1','social2','social3'] + [col for col in df.columns if 'reddit' not in col])),
        ]
    )
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = data_transformer_pipeline.fit_transform(df)
    print(df.iloc[-10:])

if __name__ == '__main__':
    main()
