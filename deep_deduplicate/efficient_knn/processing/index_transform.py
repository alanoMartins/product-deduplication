from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IndexTransform(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.names = y

        return self

    def transform(self, X) -> pd.DataFrame:

        transformed = []
        for feature in X:
            results = dict()
            for key, values in feature.items():
                results[self.names[key]] = values
            transformed.append(results)

        return transformed
