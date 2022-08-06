import re
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import gc

from efficient_knn.processing.data_generator import DataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str], model_path: str, image_path: str):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.model_path = model_path
        self.image_path = image_path

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        WGT = self.model_path
        model = tf.keras.applications.efficientnet.EfficientNetB5(weights=WGT, input_shape=None, include_top=False,
                                                                    pooling="avg",
                                                                    drop_connect_rate=0.2)

        data = DataGenerator(X, path=self.image_path)


        image_embedding = model.predict(data)
        del (model)
        gc.collect()
        return image_embedding
