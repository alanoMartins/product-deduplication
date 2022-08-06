import cv2
import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, df, img_size=456, batch_size=32, path=''):
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        self.path = path
        self.indexes = np.arange(len(self.df))

    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = len(self.df) // self.batch_size
        ct += int(((len(self.df)) % self.batch_size) != 0)
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(indexes)
        return X

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        X = np.zeros((len(indexes), self.img_size, self.img_size, 3), dtype='float32')
        df = self.df.iloc[indexes]
        for i, (index, row) in enumerate(df.iterrows()):
            img = cv2.imread(self.path + row['image'])
            X[i,] = cv2.resize(img, (self.img_size, self.img_size))
        return X