import os
import sys
import random
import pickle
import numpy as np

import utils

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

cur_dir = os.path.dirname(os.path.abspath(__file__))

class Demo():
    def __init__(self, tk_file='tokenizer.pkl', max_length=34, checkpoint='flickr_inception_transformer', config='flickr_inception_transformer'):
        tk_file = os.path.join(cur_dir, 'data/' + tk_file)
        self.tokenizer = pickle.load(open(tk_file, 'rb'))
        self.max_length = max_length

        cfg_file = os.path.join(cur_dir, 'config/' + config + '.cfg')
        self.cfg =  utils.load_cfg(cfg_file)['default']

        self.model = utils.load_model(config)

    def extract_features(self, img_file):
        if self.cfg['backbone'] == 'vgg':
            tg_size = 224
            model = VGG19()
            model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        elif self.cfg['backbone'] == 'inception':
            tg_size = 299
            model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
            model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

        img = image.load_img(img_file, target_size=(tg_size, tg_size))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        return feature

    def generate_caption(self, img_file):
        features = self.extract_features(img_file)
        if self.cfg['model'] == 'simple':
            return utils.generate_desc(self.model, self.tokenizer, features, self.max_length)
        elif self.cfg['model'] == 'transformer':
            return utils.generate_transformer_desc(self.model, self.tokenizer, features, self.max_length)

if __name__ == "__main__":
    '''
    Demo
    @args: config_name [img_file_path]
    @return: caption
    '''
    print('<Demo>')
    base_path = os.path.join(cur_dir, 'data/Flicker8k_Dataset')
    dm = Demo()
    dm = Demo(config=sys.argv[1])
    if len(sys.argv) > 2:
        file = sys.argv[2]
    else:
        file = random.choice(os.listdir(base_path))
        file = os.path.join(base_path, file)
    print('<File>:%s' % file)
    caption = dm.generate_caption(file)
    f_caption = utils.array_to_str(np.array(caption.split())[1:-1])
    print('<Caption>:%s' % f_caption)
    print('<Done>')