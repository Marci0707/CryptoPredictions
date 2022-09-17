from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from _preprocessing import CryptoCompareReader, drop_columns_deemed_as_useless, ColumnLogTransformer, \
    get_not_used_columns, ColumnDropper, DiffTransformer, ManualFeatureEngineer, TransformerWrapper


def main():
    reader = CryptoCompareReader('btc', '../cryptoCompareData', drop_na_subset=['close'], add_time_columns=True,
                                 drop_last=True)
    df = reader.read()
    not_used_columns = get_not_used_columns(df)

    take_log_columns = ['new_addresses', 'active_addresses', 'transaction_count',
                        'average_transaction_value', 'high', 'low', 'close', 'BTCTradedToUSD', 'USDTradedToBTC']
    to_scale_columns = [col for col in df.columns if col not in ['time','dayOfTheWeek','monthOfTheYear']+not_used_columns]


    column_transformer_pipeline = Pipeline(
        [
            ('column_dropper', ColumnDropper(not_used_columns)),
            ('manual_feature_engineer',ManualFeatureEngineer()),
            ('diff_taker', DiffTransformer(['block_height','current_supply'])),
            ('log_taker', ColumnLogTransformer(take_log_columns,add_one=True)),
            ('scaler',TransformerWrapper(StandardScaler(),to_scale_columns)),
            # ('pca_social', DataFrameMapper(
            #     [
            #         (df.filter(regex='reddit').columns.tolist(), PCA(n_components=0.9))
            #     ]
            #     ,df_out=True
            # )
            #  )

        ]
    )
    df = column_transformer_pipeline.fit_transform(df)
    print(df.columns)



if __name__ == '__main__':
    main()
