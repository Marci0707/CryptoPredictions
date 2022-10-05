import os
from typing import Sequence, Optional, List, Tuple, Union, Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler





def inverse_scaler_subset(windowed_data: np.ndarray, scaler: StandardScaler, subset_to_inv_transform: Sequence[str],
                          arr_col_names: Sequence[str]):


    for col_to_inv_tr in subset_to_inv_transform:
        col_idx_in_features = arr_col_names.index(col_to_inv_tr)
        col_idx_in_scaler = list(scaler.feature_names_in_).index(col_to_inv_tr)

        mean = scaler.mean_[col_idx_in_scaler]
        var = scaler.var_[col_idx_in_scaler]


        windowed_data[:,:,col_idx_in_features] = windowed_data[:,:,col_idx_in_features] * var + mean

    return windowed_data



def discretize(arr: np.ndarray, class_borders: Sequence[float]):
    # TODO more optimal
    def find_class(arr):
        classes = np.zeros(shape=(len(arr), 1))
        for value_idx, value in enumerate(arr):

            above_last_border = True
            for border_idx, border in enumerate(class_borders):
                if value < border:
                    classes[value_idx, 0] = border_idx
                    above_last_border = False
            if above_last_border:
                classes[value_idx, 0] = len(class_borders)
        return classes

    return np.apply_along_axis(find_class, -1, arr)


# used previously drop_columns_deemed_as_useless() but didnt wanna break older code
def get_not_used_columns(df) -> List[str]:
    twitter_cols = df.filter(regex='twitter').columns.tolist()
    facebook_cols = df.filter(regex='fb_').columns.tolist()
    cryptocompare_columns = ['comments', 'posts', 'followers']  # site data
    almost_duplicates = ['open']  # same as previous day close
    all_time_columns = df.filter(regex='all_time').columns.tolist()  # we have delta columns from these as well

    consequence_columns = ['hashrate', 'difficulty', 'block_time',
                           'block_size']  # these are direct consequences of transactions

    return twitter_cols + facebook_cols + cryptocompare_columns + almost_duplicates + all_time_columns + consequence_columns


def drop_columns_deemed_as_useless(df: pd.DataFrame):
    columns = get_not_used_columns(df)
    return df.drop(columns=columns)


class CryptoCompareReader:

    def __init__(self, crypto_name, folder: str, drop_na_subset: Optional[Sequence[str]] = None, add_time_columns=True,
                 drop_last=False):
        self.crypto_name = crypto_name
        self.folder = folder
        self.drop_na_subset = drop_na_subset
        self.add_time_columns = add_time_columns
        self.drop_last = drop_last

        self.price_columns = None
        self.blockchain_data_columns = None
        self.social_columns = None

    def _add_time_columns(self, all_info: pd.DataFrame) -> pd.DataFrame:
        all_info['time'] = pd.to_datetime(all_info['time'])
        all_info['time'] = pd.to_datetime(all_info['time'])
        all_info['dayOfTheWeek'] = all_info['time'].dt.dayofweek
        all_info['monthOfTheYear'] = all_info['time'].dt.month

        return all_info

    def read(self) -> pd.DataFrame:
        prices = pd.read_csv(os.path.join(self.folder, f"{self.crypto_name}_prices.csv"), thousands=',')
        blockchain_data = pd.read_csv(os.path.join(self.folder, f"{self.crypto_name}_blockchain_data.csv"),
                                      thousands=',')
        social = pd.read_csv(os.path.join(self.folder, f"{self.crypto_name}_social.csv"), thousands=',')

        self.price_columns = prices.drop(columns='time').columns.tolist()
        self.blockchain_data_columns = blockchain_data.drop(columns='time').columns.tolist()
        self.social_columns = social.drop(columns='time').columns.tolist()

        all_info = prices \
            .merge(blockchain_data, on='time', how='outer') \
            .merge(social, on='time', how='outer') \
            .sort_values('time') \
            .reset_index(drop=True)
        if self.drop_na_subset:
            all_info.dropna(subset=self.drop_na_subset, inplace=True)

        if self.add_time_columns:
            all_info = self._add_time_columns(all_info)

        all_info.reset_index(drop=True, inplace=True)

        if self.drop_last:
            all_info = all_info.iloc[:-1]

        return all_info


class ColumnLogTransformer(TransformerMixin):

    def __init__(self, columns: Union[List[str], str] = None, only_from_first_nonzero: Union[List[str], str] = None,
                 add_one=False):
        self.columns = columns
        self.only_from_first_nonzero = only_from_first_nonzero
        self.add_one = add_one

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # to avoid log(0)
        shift = 1 if self.add_one else 0

        if self.columns is None:
            X = np.log(X + shift)
        else:
            X[self.columns] = np.log(X[self.columns] + shift)
        return X


class ColumnDropper(TransformerMixin):

    def __init__(self, columns: Union[List[str], str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.drop(columns=self.columns, inplace=True)
        return X


class Pandanizer(TransformerMixin):
    """for sklearn transformers whose outputs are numpy arrays"""

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(data=X, columns=self.columns)

        return df


class PCAFromFirstValidIndex(PCA):

    def fit(self, X, y=None):
        first_valid_index = X.first_valid_index()
        return super().fit(X.iloc[first_valid_index:], y)

    def fit_transform(self, X, y=None):
        first_valid_index = X.first_valid_index()

        pca_result = super().fit_transform(X.iloc[first_valid_index:], y)

        pca_result = pd.DataFrame(data=pca_result, columns=list(range(pca_result.shape[1])))

        padding = pd.DataFrame(pd.NA, index=list(range(first_valid_index)), columns=list(range(pca_result.shape[1])))
        concat_res = pd.concat([padding, pca_result], axis=0)
        return concat_res


class ManualFeatureEngineer:

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # there is a week (2011-jun.20-25) where the traded value is strangely 0.0 for all days (see insights.ipynb):
        X.loc[X["BTCTradedToUSD"] == 0.0, 'BTCTradedToUSD'] = (X["BTCTradedToUSD"].shift(7) + X["BTCTradedToUSD"].shift(
            -7)) / 2.0

        # want to reduce dimensions
        X['daily_movement'] = X['high'] - X['low']
        # X.drop(columns=['high', 'low'], inplace=True)

        # this way taking log is not needed. there are a lot of zeros.
        # it matches the distribution of the large_transaction_count (see feature_engineering.ipynb)
        X['large_transaction_count'] = X['large_transaction_count'] / X['transaction_count']

        return X


class DiffTransformer(TransformerMixin):

    def __init__(self, columns: Union[List[str], str] = None, replace_nan_to_zeros=False):
        self.columns = columns
        self.replace_nan_to_zeros = replace_nan_to_zeros  # future transformers cannot always handler nan values in first row

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if self.columns:
            X[self.columns] = X[self.columns].pct_change()
        else:
            X = X.pct_change()

        if self.replace_nan_to_zeros:
            X = X.replace([np.nan,np.inf,-np.inf], 0.0)
        return X


class LinearCoefficientTargetGenerator(TransformerMixin):

    def __init__(self, source_column_name: str, regression_for_days_ahead: int,window_size:int,
                 result_column_name: str = 'LinearCoeffTarget_target',
                 classifier_borders: Optional[Union[Tuple[float, float], Tuple[float]]] = None):
        self.source_column_name = source_column_name
        self.for_days_ahead = regression_for_days_ahead
        self.window_size = window_size
        self.result_column_name = result_column_name
        self.classifier_borders = classifier_borders

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):

        lm = LinearRegression(fit_intercept=False)

        def calc_coeffs(x):

            if np.std(x) == 0:
                return 0

            scaled = (x.to_numpy() - x.iloc[0]) / np.std(x)
            try:
                lm.fit(np.array(list(range(len(scaled)))).reshape(-1, 1), scaled)
            except ValueError as e:
                print(scaled)
                raise e

            return lm.coef_[0]

        series = X[self.source_column_name].rolling(self.for_days_ahead).apply(lambda x: calc_coeffs(x))

        # at the end we cannot compute the full regression for window_size days ahead because there is not enough timestamp for the future
        target = series.copy(deep=True)
        feature = series.copy(deep=True) #this can remain unchanged because its the regression of the past n days
        target.iloc[:self.window_size+self.for_days_ahead-2] = pd.NA #the target of the first 14 days is the regression for 14:24 days
        target = target.iloc[self.for_days_ahead-1:].reset_index(drop=True) #slide it back so the first target will be in the 14th row (if winsize=14)
        target = pd.concat([target,pd.Series([pd.NA]*self.for_days_ahead)],axis=0,ignore_index=True) # we dont know the regression for the future beyond the data

        if self.classifier_borders:  # TODO generalize for n target
            # classes :  [-1,0,1] for [decrease,stationary,increase]
            if len(self.classifier_borders) == 2:

                target = target.apply(lambda value:
                                                    pd.NA if pd.isna(value) else
                                                    0 if value < self.classifier_borders[0] else
                                                    2 if value > self.classifier_borders[1]
                                                    else 1)
            # classes [-1,1] for down or up
            elif len(self.classifier_borders) == 1:

                target = target.apply(lambda value:
                                                    pd.NA if pd.isna(value) else
                                                    0 if value < self.classifier_borders[0]
                                                    else 1)

            else:
                raise ValueError(f'invalid class borders {self.classifier_borders}')


        X[self.result_column_name] = target
        X[self.result_column_name.replace('target','feature')] = feature

        return X


class FutureDayTargetGenerator(TransformerMixin):

    def __init__(self, source_column_name: str, for_days_ahead: int, result_column_name: str = 'FutureDayTarget'):
        self.source_column_name = source_column_name
        self.for_days_ahead = for_days_ahead
        self.result_column_name = result_column_name

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        X[self.result_column_name] = X[self.source_column_name].shift(self.for_days_ahead)
        return self


class SmoothedDerivativesGenerator(TransformerMixin):

    def __init__(self, source_col: str, window_size: Optional[int] = None, halflife: Optional[int] = None,
                 n_derivatives: int = 2,
                 column_prefix_name: str = 'smoothed_derivative', smoothing_type='rolling_mean'):

        if smoothing_type not in ('ewm', 'rolling_mean'):
            raise Exception(
                'Only "rolling_mean" for rolling mean and exponentially weighted averages, "ewm" are supported')

        if (window_size is None and halflife is None) or (window_size is not None and halflife is not None):
            raise Exception('Provide window size for rolling_mean or halflife for exponential')

        if n_derivatives <= 0:
            raise Exception('Derivative count must be positive')

        self.source_col = source_col
        self.smoothing_param = window_size if window_size is not None else halflife
        self.n_derivatives = n_derivatives
        self.column_prefix_name = column_prefix_name
        self.smoothing_type = smoothing_type

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):

        if self.smoothing_type == 'ewm':
            smoothed_series = X[self.source_col].ewm(halflife=self.smoothing_param).mean()
        else:  # self.smoothing_type == 'rolling_mean' hopefully:
            smoothed_series = X[self.source_col].rolling(self.smoothing_param).mean()

        n_derivates = smoothed_series - smoothed_series.shift(-1)

        X[f"{self.column_prefix_name}_1"] = n_derivates

        for derivative_idx in range(1, self.n_derivatives):
            column_name = f"{self.column_prefix_name}_{derivative_idx + 1}"
            X[column_name] = n_derivates - n_derivates.shift(-1)

        return X


class ManualValidTargetDetector(TransformerMixin):

    def __init__(self, invalidate_top_x_percent,window_size:int, regression_days:int):
        self.invalidate_top_x_percent = invalidate_top_x_percent
        self.window_size = window_size
        self.regression_days = regression_days

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        #filtering based on spikes in social media and prices
        positive_reddit_change = X['reddit_comments_per_day'].diff().clip(lower=0)
        price_change = np.abs(X['close'].pct_change())

        result = positive_reddit_change.fillna(0) + 2 * price_change.fillna(0)

        tops = result.sort_values(ascending=False)
        invalid_n_rows = int(len(X) * self.invalidate_top_x_percent / 100.0)

        invalid_indices = tops.iloc[:invalid_n_rows].index

        series = X.apply(lambda row: 1 if row.name not in invalid_indices else -1, axis=1)

        #filtering based on architecture
        series.iloc[:self.window_size-1] = -1 #cannot predict if there is not enough data behind
        series.iloc[-self.regression_days:] = -1 #cannot predict if there is not enough data ahead

        X['is_valid_target'] = series
        return X


class WindowGenerator(TransformerMixin):

    def __init__(self, window_size: int, features: Sequence[str], targets: Sequence[str],
                 is_valid_target_col_name: str):
        self.is_valid_target_col_name = is_valid_target_col_name
        self.window_size = window_size
        self.features = features
        self.targets = targets
        self.banned_indices = []

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> Tuple[np.ndarray, np.ndarray]:
        banned_target_indices = X.loc[X[self.is_valid_target_col_name] < 0].index

        self.banned_indices = list(banned_target_indices)

        indices = np.array(range(len(X)))

        windows = sliding_window_view(indices, self.window_size)  # last one is the target
        filtered_windows = np.array([window for window in windows if window[-1] not in banned_target_indices])

        x_data = []
        y_data = []
        for window in filtered_windows:
            df_part = X.iloc[window]
            features = df_part.iloc[:-1][self.features]
            targets = df_part.iloc[-1][self.targets]

            x_data.append(features.to_numpy())
            y_data.append(targets.to_numpy())

        return np.array(x_data), np.array(y_data)
