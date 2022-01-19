import os
import sys
import random
import pickle
import numpy as np

import utils
import transformers_tf.model as model
import transformers_tf.data_generator as data_gen

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

cur_dir = os.path.dirname(os.path.abspath(__file__))

class Demo():
    def __init__(self, tk_file='tokenizer.pkl', max_length=34, config='flickr_inception_transformer'):
        tk_file = os.path.join(cur_dir, 'data/' + tk_file)
        self.tokenizer = pickle.load(open(tk_file, 'rb'))
        self.chk_dir = os.path.join(cur_dir, 'checkpoints/' + config)
        self.chk_file = os.path.join(self.chk_dir, config + '.h5')

        cfg_file = os.path.join(cur_dir, 'config/' + config + '.cfg')
        self.cfg_name = config
        self.cfg =  utils.load_cfg(cfg_file)['default']
        
        self.max_length = max_length
        self.vocab_size = int(self.cfg['vocab_size'])
        self.model = utils.load_model(config)

    def extract_features(self, img_file):
        if self.cfg['backbone'] == 'vgg':
            tg_size = 224
            model = VGG19()
            model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        elif self.cfg['backbone'] == 'inception':
            tg_size = 299
            if any(char.isdigit() for char in self.cfg['model']):
                model = InceptionV3(weights='imagenet', include_top=False)
            else:
                model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
            model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

        img = image.load_img(img_file, target_size=(tg_size, tg_size))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        return feature

    def generate_caption(self, img_file):
        # Not loading weights???
        features = self.extract_features(img_file)
        if self.cfg['model'] == 'simple':
            return utils.generate_desc(self.model, self.tokenizer, features, self.max_length)
        elif self.cfg['model'] == 'transformer':
            return utils.generate_transformer_desc(self.model, self.tokenizer, features, self.max_length)
        elif self.cfg['model'] == 'transformer2d':
            features = np.reshape(features, (features.shape[0], -1, features.shape[3]))
            return self.generate_caption2d(features)

    def generate_caption2d(self, features):
        import tensorflow as tf    
        caption_model = utils.load_model(self.cfg_name)
        test_gen = data_gen.DataGenerator(self.vocab_size, self.max_length, self.cfg_name, train=False)

        # Need to build model before loading weights
        f_tensor, seq_tensor = test_gen.__getitem__(0)
        pad_mask, look_mask = model.create_masks(seq_tensor)
        comb_mask = tf.maximum(pad_mask, look_mask)

        _, _ = caption_model(f_tensor, seq_tensor, False, comb_mask)
        tmp_file = self.chk_file.split('/')[-1]
        print('Loading {} weights...'.format(tmp_file))
        caption_model.load_weights(self.chk_file)
        # # # Built and loaded weights
        pred, _, _ = utils.generate_transformer2d_desc(caption_model, self.tokenizer, features, self.max_length)
        return pred

if __name__ == "__main__":
    '''
    Demo
    @args: config_name [img_file_path]
    @return: caption
    '''
    print('<Demo>')
    base_path = os.path.join(cur_dir, 'data/Flicker8k_Dataset')
    dm = Demo(config=sys.argv[1])
    if len(sys.argv) > 2:
        file = sys.argv[2]
    else:
        file = random.choice(os.listdir(base_path))
        file = os.path.join(base_path, file)
    print('<File>:%s' % file)
    caption = dm.generate_caption(file)
    #f_caption = utils.array_to_str(np.array(caption.split())[1:-1])
    print('<Caption>:%s' % caption)
    print('<Done>')