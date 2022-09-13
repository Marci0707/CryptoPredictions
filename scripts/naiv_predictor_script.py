from _preprocessing import CryptoCompareReader, LinearCoefficientTargetGenerator, drop_columns_deemed_as_useless


def main():
    reader = CryptoCompareReader('btc', '../cryptoCompareData', drop_na_subset=['close'], add_time_columns=True,drop_last=True)
    df = reader.read()
    gen = LinearCoefficientTargetGenerator(source_column_name='close',
                                          regression_for_days_ahead=14)

    df = drop_columns_deemed_as_useless(df)

    df = gen.fit_transform(df)

    print(df)

if __name__ == '__main__':
    main()
