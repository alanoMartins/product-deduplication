import re
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SanitizeText(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.CM_MAP = {"wanita": "woman", "anak": "child", "bayi": "baby", "tas": "bag", "masker": "face mask",
                       "pria": "men", "murah": "cheap", "tangan": "hand", "alat": "tool", "motif": "motive",
                       "warna": "color", "bahan": "material", "celana": "pants", "baju": "clothes", "kaos": "t-shirt",
                       "sepatu": "shoes", "rambut": "hair", "mainan": "toy", "sarung": "holster", "polos": "plain",
                       "rak": "rack", "botol": "bottle", "sabun": "soap", "kain": "fabric", "panjang": "long",
                       "kabel": "cable", "buku": "book", "plastik": "plastic", "mobil": "car", "hitam": "black",
                       "karakter": "character", "putih": "white", "dompet": "purse", "kaki": "feet",
                       "pembersih": "cleaners", "lipat": "folding", "silikon": "silicone", "minyak": "oil",
                       "isi": "contents", "paket": "package", "susu": "milk", "gamis": "robe", "mandi": "bath",
                       "madu": "honey", "kulit": "skin", "serbaguna": "multipurpose", "bisa": "can",
                       "kacamata": "spectacles", "pendek": "short", "tali": "rope", "selempang": "sash", "topi": "hat",
                       "obat": "drug", "gantungan": "hanger", "tahun": "year", "jilbab": "hijab", "dapur": "kitchen",
                       "dinding": "wall", "kuas": "brush", "perempuan": "woman", "katun": "cotton", "sepeda": "bike",
                       "lucu": "funny", "lengan": "arm", "kaca": "glass", "garansi": "warranty", "bunga": "flower",
                       "handuk": "towel", "dewasa": "adult", "elektrik": "electric", "timbangan": "balance",
                       "besar": "big", "bahan": "ingredient", "ransel": "backpack", "kertas": "paper", "lampu": "light",
                       "sepatu": "shoes", "tempat": "place"}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        def remove_patterns(text):
            '''
            Some titles contain " x " as separators.
            Example: train_3369186413
            '''
            patterns = {' x ': ' ',
                        ' a ': ' '}
            for k, v in patterns.items():
                text = text.replace(k, v)
            return text

        # so that we do not over-write the original dataframe
        X = X.copy()

        # letters_only = lambda text: re.sub("[^a-zA-Z]", " ", text)
        lowercase_only = lambda text: text.lower()
        replace_multispace_by_space = lambda text: re.sub('\s+', ' ', text)
        translate_ind_to_eng = lambda text: " ".join(self.CM_MAP.get(w, w) for w in text.split())

        for feature in self.variables:
            X[feature] = (X[feature]
                          .apply(lowercase_only)
                          .apply(remove_patterns)
                          .apply(replace_multispace_by_space)
                          .apply(translate_ind_to_eng))

            return X[feature]
