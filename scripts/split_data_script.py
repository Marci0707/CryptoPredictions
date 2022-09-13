from _preprocessing import CryptoCompareReader


def main():

    split_at_year = 2022

    reader = CryptoCompareReader('btc','../cryptoCompareData',drop_na_subset=['close'],add_time_columns=True,drop_last=True)
    df = reader.read()

    df_train = df.loc[df.time.dt.year < split_at_year].reset_index(drop=True)
    df_test = df.loc[df.time.dt.year >= split_at_year].reset_index(drop=True)

    df_train.to_csv('../data/training_set.csv')
    df_test.to_csv('../data/test_set.csv')



if __name__ == '__main__':
    main()