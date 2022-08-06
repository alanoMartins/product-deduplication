from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CombinePredictionsTransform(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X) -> pd.DataFrame:
        X.reset_index()
        results = [self.combine_multi_predictions(row) for row in X.iterrows()]

        X["pred_combined"] = results

        return X

    def combine_multi_predictions(self, row):
        idt = row[1]["posting_id"]
        pred_text = row[1]['text_pred']
        pred_img = row[1]['image_pred']

        pred_text_keys = pred_text.keys()
        pred_img_keys = pred_img.keys()

        result = dict()

        for key in pred_text_keys:

            if key in result:
                continue

            if key == idt:
                continue
            value_text = 1 - pred_text[key] if key in pred_text else 0
            value_img = 1 - pred_img[key] if key in pred_img else 0

            total = value_text + value_img

            result[key] = total

        for key in pred_img_keys:

            if key in result:
                continue

            if key == idt:
                continue

            value_text = 1 - pred_text[key] if key in pred_text else 0
            value_img = 1 - pred_img[key] if key in pred_img else 0

            total = value_text + value_img

            result[key] = total

        return str(result)
