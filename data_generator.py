import os
import pickle
import numpy as np
import tensorflow as tf

import utils

cur_dir = os.path.dirname(os.path.abspath(__file__))

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, vocab_size, max_length, cfg='flickr_inception_transformer', shuffle=True, train=True, load=True):
        self.descs_file = os.path.join(cur_dir, 'data/descriptions.txt')
        tk_file = os.path.join(cur_dir, 'data/tokenizer.pkl')
        
        cfg_file = os.path.join(cur_dir, 'config/' + cfg + '.cfg')
        self.cfg =  utils.load_cfg(cfg_file)['default']
        # Transformer2d
        if any(char.isdigit() for char in self.cfg['model']):
            self.img_file = os.path.join(cur_dir, 'data/features_' + self.cfg['backbone'] + '2d.pkl')
        else:
            self.img_file = os.path.join(cur_dir, 'data/features_' + self.cfg['backbone'] + '.pkl')

        self.batch_size = int(self.cfg['batch_size'])
        self.shuffle = shuffle

        if train:
            file = os.path.join(cur_dir, 'data/Flickr_8k.trainImages.txt')
        else:
            file = os.path.join(cur_dir, 'data/Flickr_8k.devImages.txt')
        self.ids = utils.load_set(file)
        self.maxlen = max_length

        if train and not load:
            self.tokenizer = utils.set_tokenizer(self.ids)
            utils.save_tokenizer(self.tokenizer, tk_file)
        else:
            self.tokenizer = pickle.load(open(tk_file, 'rb'))

        self.model = self.cfg['model']
        self.vocab_size = vocab_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_tmp):
        features = utils.load_img_features(self.img_file, ids_tmp)
        descs = utils.load_clean_descriptions(self.descs_file, ids_tmp)
        if self.model == 'transformer' or self.model == 'transformer2d':
            return utils.create_tensor_seqs(self.tokenizer, self.maxlen, descs, features, self.vocab_size)
        else:
            return utils.create_all_seqs(self.tokenizer, self.maxlen, descs, features, self.vocab_size)

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1)*self.batch_size]
        tmp = list(self.ids)
        ids = [tmp[k] for k in indexes]

        x1, x2, y = self.__data_generation(ids)
        if len(x1.shape) > 3:
            x1 = np.reshape(x1, (x1.shape[0], -1, x1.shape[3]))
        #print('NEW SHAPES: x1>{}, x2>{}, y>{}'.format(x1.shape, x2.shape, y.shape))
        return x1, x2, y

    def get_batch_size(self):
        return self.batch_size

    def get_max_count(self):
        return len(self.ids)