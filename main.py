import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from crypto_models import *
from evaluation import evalute_model


def read_and_merge_data():
    #reead
    btc_usd = pd.read_csv("data/BTC_USD Binance Historical Data.csv", thousands=',')[::-1]
    eth_usd = pd.read_csv("data/ETH_USD Binance Historical Data.csv", thousands=',')[::-1]

    eur_usd = pd.read_csv("data/EUR_USD Historical Data.csv", thousands=',')[::-1]
    gold_usd = pd.read_csv("data/XAU_USD Historical Data.csv", thousands=',')[::-1]
    real_estate_usd = pd.read_csv("data/Real Estate Historical Data.csv", thousands=',')[::-1]
    nyse_usd = pd.read_csv("data/NYSE Composite Historical Data.csv", thousands=',')[::-1]


    #merge into 1 df
    raw_data = (btc_usd, eth_usd, eur_usd, gold_usd, real_estate_usd, nyse_usd)
    exchange_rates_names = ["BTC", "ETH", "EUR", "Gold", "RealEstate", "NYSE"]
    cryptos = ["BTC", "ETH"]
    column_of_interest = 'Open'

    for df in raw_data:
        df["Date"] = pd.to_datetime(df["Date"])

    mutual = None

    for idx in range(0, len(raw_data)):
        right_df = raw_data[idx]

        unique_name = f"{column_of_interest}_{exchange_rates_names[idx]}"

        right_df.rename(columns={column_of_interest: unique_name}, inplace=True)
        right_df.drop(columns=[col for col in right_df.columns if col not in ["Date", unique_name]], inplace=True)

        if mutual is None:
            mutual = right_df
        else:
            mutual = mutual.merge(right=right_df, on="Date", how="outer")

    mutual.sort_values(by="Date", inplace=True)
    mutual.reset_index(inplace=True, drop=True)

    #missing values
    for col in mutual.columns:
        if col.split("_")[-1] in cryptos:
            mutual[col].fillna(0, inplace=True)
        elif col != "Date":
            mutual[col] = pd.to_numeric(mutual[col]).astype(np.float64)
            mutual[col].interpolate(method='polynomial', order=2, inplace=True)
    mutual.dropna(inplace=True)
    mutual = mutual[mutual["Open_BTC"] != 0]
    mutual.reset_index(inplace=True, drop=True)

    #convert date
    mutual["day_number"] = mutual.index
    if "Date" in mutual.columns:
        mutual.drop("Date", inplace=True, axis=1)

    return mutual


def split_data(df : pd.DataFrame, window_size : int, test_ratio : int):

    x_cols = ["Open_BTC", "Open_ETH", "Open_EUR", "Open_Gold", "Open_RealEstate", "Open_NYSE", "day_number"]
    y_cols = ["Open_BTC"]



    y_data = df[y_cols].to_numpy()
    x_data = df[x_cols].to_numpy()

    features = x_data.shape[-1]

    windows = sliding_window_view(x_data, window_shape=(window_size, features))
    windows = windows.reshape( (windows.shape[0], windows.shape[2], windows.shape[3]) )

    targets = y_data[window_size:]
    windows = windows[:-1]

    separator_index = round(len(windows) * test_ratio)
    x_train = windows[:separator_index]
    y_train = targets[:separator_index]

    x_test = windows[separator_index:]
    y_test = targets[separator_index:]

    return x_train, y_train, x_test, y_test

def main():

    df = read_and_merge_data()

    X_train, Y_train, X_test, Y_test = split_data(df, window_size=10, test_ratio = 0.2)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    Y_train = y_scaler.fit_transform(Y_train)

    X_test = x_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    Y_test = y_scaler.transform(Y_test)

    last_value_predictor = LastValuePredictor()

    fccn_model = get_fccn_baseline(input_shape=(X_train.shape[-2],X_train.shape[-1]))
    fccn_model.compile(optimizer='Adam', loss='mse')

    history = fccn_model.fit(X_train, Y_train, epochs=10, validation_split=0.2)

    print(X_train.shape)
    gru_model = get_gru_baseline(input_shape=(X_train.shape[-2],X_train.shape[-1]))
    gru_model.compile(optimizer='Adam', loss='mse')

    history = gru_model.fit(X_train, Y_train, epochs=15, validation_split=0.2, batch_size=32)


    evalute_model(last_value_predictor,X_test,Y_test,y_scaler, model_name = "last_value",only_last_values=100)
    evalute_model(fccn_model,X_test,Y_test,y_scaler, model_name = "fccn",only_last_values=50)
    evalute_model(gru_model,X_test,Y_test,y_scaler, model_name = "gru",only_last_values=50)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
