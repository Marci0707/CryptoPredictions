from typing import Union
import random

import numpy as np


class LastValuePredictor:

    def __init__(self, predict_feature_idx: int):
        self.feature_idx = predict_feature_idx

    def predict(self, X: np.ndarray):
        """X shape: [samples, timesteps, features]"""
        return X[:, :, self.feature_idx]


class RandomBinaryPredictor:

    def __init__(self, predict_feature_idx: int, match_distribution=True):
        self.feature_idx = predict_feature_idx
        self.match_distribution = match_distribution
        self.guess_distribution = np.array([0.5, 0.5])

    def predict(self, X: np.ndarray):
        """X shape: [samples, timesteps, features]"""

        arr = X[:, -1, self.feature_idx].flatten()
        values, counts = np.unique(arr, return_counts=True)

        if self.match_distribution:
            # returns the values as sorted so 0 will be at the first index

            self.guess_distribution = counts / len(arr)

        return random.choices(population=values, weights=self.guess_distribution,k=len(arr))
