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

    social_columns = [col for col in reader.social_columns if col in df.columns]#['reddit_subscribers', 'reddit_active_users', 'reddit_posts_per_hour', 'reddit_posts_per_day', 'reddit_comments_per_hour', 'reddit_comments_per_day'] 6
    prices_columns = [col for col in reader.price_columns if col in df.columns] #['new_addresses', 'active_addresses', 'transaction_count', 'large_transaction_count', 'average_transaction_value', 'block_height', 'hashrate', 'difficulty', 'block_time', 'block_size', 'current_supply'] 11
    blockchain_data_columns = [col for col in reader.blockchain_data_columns if col in df.columns] #['time', 'dayOfTheWeek', 'monthOfTheYear'] 3



if __name__ == '__main__':
    main()