from typing import *

import numpy as np
from joblib import dump, load
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class GaussianProcess:
    def __init__(self):
        self.model = GaussianProcessRegressor(
            kernel=Matern(),
            n_restarts_optimizer=3,
            random_state=42
        )
        self.training_points: List[np.ndarray, np.ndarray] = [np.empty((0, 0)), np.empty((0, 0))]

    def set_training_data(self, points: np.ndarray, values: np.ndarray) -> None:
        if points.shape[0] != values.shape[0]:
            raise ValueError("Variables 'points' and 'values' must have the same first dimension")

        x = points.copy()
        y = values.copy()
        self.training_points = [x, y]

    def update_training_values(self, values: np.ndarray) -> None:
        # if self.training_points[0].shape[0] != values.shape[0]:
        #     raise ValueError("Variables 'points' and 'values' must have the same first dimension")

        y = values.copy()
        self.training_points[1] = y

    def train(self) -> None:
        # make copy of training points
        X, Z = self.copy_data()
        self.model.fit(X, Z)  # fit model to training points

    def predict(self, X: np.ndarray) -> np.ndarray:
        # if X.shape[1] != self.training_points[0].shape[1]:
        #     raise ValueError("Given points are not correct size")
        if X.ndim == 1: X = X.reshape(1, -1)
        return self.model.predict(X)  # return predictions

    def copy_data(self):
        X = self.training_points[0].copy()
        Z = self.training_points[1].copy()
        return X, Z

    def save_model(self, filename: str):
        if '.joblib' not in filename:
            filename += '.joblib'
        dump(self.model, filename)

    def load_model(self, filename):
        self.model = load(filename)
