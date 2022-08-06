import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors


class DistanceEstimator(BaseEstimator, RegressorMixin):

    def __init__(self, threshold):
        self.threshold = threshold
        self.model = NearestNeighbors(n_neighbors=6, metric="cosine")

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.model.fit(X)
        self.names = y
        return self

    def predict(self, X: pd.DataFrame):
        chunk = 10
        cl = len(self.names) // chunk
        cl += int((len(self.names) % chunk) != 0)
        pred_img = []
        for i in range(cl):

            a = i * chunk
            b = (i + 1) * chunk
            b = min(len(self.names), b)
            distances, indices = self.model.kneighbors(X[a:b, ])
            for j in range(b - a):
                distance = distances[j, :]
                ind = np.where(distance < self.threshold)[0]
                IND = indices[j, ind]

                near_distance = distance[ind]

                result = dict()
                for dist, idx in zip(near_distance, IND):
                    result[idx] = dist

                pred_img.append(result)

        return pred_img
