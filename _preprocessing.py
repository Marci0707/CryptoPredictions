import os
from typing import Sequence, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import TransformerMixin


class CryptoCompareReader:
    # TODO drop last
    def __init__(self, crypto_name, folder: str, drop_na_subset: Optional[Sequence[str]] = None, add_time_columns=True):
        self.crypto_name = crypto_name
        self.folder = folder
        self.drop_na_subset = drop_na_subset
        self.add_time_columns = add_time_columns

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

        return all_info


class LinearCoefficientTargetGenerator(TransformerMixin):

    def __init__(self, source_column_name: str,window_size:int, regression_for_days_ahead: int, result_column_name: str = 'LinearCoeffTarget'):
        self.source_column_name = source_column_name
        self.window_size = window_size
        self.for_days_ahead = regression_for_days_ahead
        self.result_column_name = result_column_name

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        def calc_coeffs(x):

            shifted = x.to_numpy() - x.iloc[0]

            x = np.array(list(range(len(shifted))))
            x = x[:, np.newaxis]

            slope, _, _, _ = np.linalg.lstsq(x, shifted)
            return slope

        series = X[self.source_column_name].rolling(self.window_size).apply(lambda x: calc_coeffs(x))

        # drop na values at the beggining. This drops window_size-1 elements at the begginig when the window is invalid = not full
        series.dropna(inplace=True)


        #at the end we cannot compute the full regression for window_size days ahead because there is not enough data for the future
        series = pd.concat([series,pd.Series(np.nan)*self.window_size]).reset_index(drop=True)

        X[self.result_column_name] = series

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


class WindowsGenerator(TransformerMixin):

    def __init__(self, window_size: int, features: Sequence[str], targets: Sequence[str],
                 ignore_targets_above_pct_change: Optional[List[Tuple[str, float]]] = None):
        """if window_size == 10 then 9 of the rows will be used for predicting the 10th.
        so the last row is always the target"""
        self.ignore_targets_above_pct_change = ignore_targets_above_pct_change
        self.window_size = window_size
        self.features = features
        self.targets = targets

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> Tuple[np.ndarray, np.ndarray]:

        banned_target_indices = set()
        if self.ignore_targets_above_pct_change:

            for column_name, pct in self.ignore_targets_above_pct_change:
                abs_pct_changes = abs(X[column_name].pct_change()) * 100
                idx_ls = list(X.loc[X[abs_pct_changes > pct].index].index)
                banned_target_indices.update(idx_ls)

        indices = np.array(range(len(X)))

        windows = sliding_window_view(indices, self.window_size)
        filtered_windows = np.array([window for window in windows if window[-1] not in banned_target_indices])

        x_data = []
        y_data = []
        for window in filtered_windows:
            df_part = X.iloc[window]
            features = df_part.iloc[:-1][self.features]
            targets = df_part.iloc[-1][self.targets]
            x_data.append(features.to_numpy())
            y_data.append(targets.to_numpy())

        return np.array(x_data),np.array(y_data)