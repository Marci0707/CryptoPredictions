import pandas as pd

from _preprocessing import CryptoCompareReader, drop_columns_deemed_as_useless


def main():
    split_at_year = 2022

    prices = pd.read_csv('../cryptoCompareData/btc_prices.csv')
    social = pd.read_csv('../cryptoCompareData/btc_social.csv')
    blockchain = pd.read_csv('../cryptoCompareData/btc_blockchain_data.csv')

    for name, df in {'btc_prices': prices, 'btc_social': social, 'btc_blockchain_data': blockchain}.items():
        df['time'] = pd.to_datetime(df['time'])
        df_train = df.loc[df.time.dt.year < split_at_year].reset_index(drop=True)
        df_test = df.loc[df.time.dt.year >= split_at_year].reset_index(drop=True)
        df_test.to_csv(f'../splits/test/{name}.csv')
        df_train.to_csv(f'../splits/train/{name}.csv')


if __name__ == '__main__':
    main()
