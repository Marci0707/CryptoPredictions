from typing import List, Tuple

import pandas as pd
from keras import Input
from keras.layers import Dense

from _preprocessing import CryptoCompareReader, drop_columns_deemed_as_useless


def get_branched_vae():

    social_input = Input(shape=(6,))
    chain_data_input = Input(shape=(11,))
    prices_input = Input(shape=(5,))
    others_input = Input(shape=3,)

    Dense(4)(social_input)






def main():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    reader = CryptoCompareReader('btc', '../cryptoCompareData', drop_na_subset=['close'], add_time_columns=True,drop_last=True)
    df = reader.read()

    df = drop_columns_deemed_as_useless(df)

    social_columns = [col for col in reader.social_columns if col in df.columns]
    prices_columns = [col for col in reader.price_columns if col in df.columns]
    blockchain_data_columns = [col for col in reader.blockchain_data_columns if col in df.columns]

    other_columns = [col for col in df.columns if col not in set(social_columns+prices_columns+blockchain_data_columns)]

    print(social_columns,len(social_columns)) #['reddit_subscribers', 'reddit_active_users', 'reddit_posts_per_hour', 'reddit_posts_per_day', 'reddit_comments_per_hour', 'reddit_comments_per_day'] 6
    print(prices_columns,len(prices_columns)) # ['high', 'low', 'close', 'BTCTradedToUSD', 'USDTradedToBTC'] 5
    print(blockchain_data_columns,len(blockchain_data_columns)) #['new_addresses', 'active_addresses', 'transaction_count', 'large_transaction_count', 'average_transaction_value', 'block_height', 'hashrate', 'difficulty', 'block_time', 'block_size', 'current_supply'] 11
    print(other_columns,len(other_columns)) #['time', 'dayOfTheWeek', 'monthOfTheYear'] 3



if __name__ == '__main__':
    main()