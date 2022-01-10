import tensorflow
import numpy as np

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, ids, img_dim, seq_dim, out_dim, batch_size=2, shuffle=True):
        self.ids = ids
        self.img_dim = img_dim
        self.seq_dim = seq_dim
        self.out_dim = out_dim

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_tmp):
        x1 = np.empty((self.batch_size, *self.img_dim))
        x2 = np.empty((self.batch_size, *self.img_dim))
        y = np.empty((self.batch_size), dtype=int)

        for i, id in enumerate(ids_tmp):
            #x1[i,] = ...

        return x1, x2, y

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1)*self.batch_size]
        ids = [self.ids[k] for k in indexes]

        x1, x2, y = self.__data_generation(ids)
        return x1, x2, y