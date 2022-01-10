import os
import sys
import random
import pickle
import numpy as np

import utils

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

cur_dir = os.path.dirname(os.path.abspath(__file__))

class Demo():
    def __init__(self, tk_file='tokenizer.pkl', max_length=34, checkpoint='flickr_vgg_simple', config='flickr_vgg.cfg'):
        tk_file = os.path.join(cur_dir, 'data/' + tk_file)
        self.tokenizer = pickle.load(open(tk_file, 'rb'))
        self.max_length = max_length

        ch_file = os.path.join(cur_dir, 'checkpoints/' + checkpoint)
        self.model = load_model(ch_file)

        cfg_file = os.path.join(cur_dir, 'config/' + config)
        self.cfg =  utils.load_cfg(cfg_file)

    def extract_features(self, img_file):
        if self.cfg['default']['backbone'] == 'vgg':
            model = VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        img = image.load_img(img_file, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        return feature

    def generate_caption(self, img_file):
        features = self.extract_features(img_file)
        return utils.generate_desc(self.model, self.tokenizer, features, self.max_length)

if __name__ == "__main__":
    print('<Demo>')
    base_path = os.path.join(cur_dir, 'data/Flicker8k_Dataset')
    dm = Demo()
    if len(sys.argv) > 1:
        file = sys.argv[1]
        if len(sys.argv) == 3:
            dm = Demo(config=sys.argv[2])
    else:
        file = random.choice(os.listdir(base_path))
        file = os.path.join(base_path, file)
    print('<File>:%s' % file)
    caption = dm.generate_caption(file)
    f_caption = utils.array_to_str(np.array(caption.split())[1:-1])
    print('<Caption>:%s' % f_caption)
    print('<Done>')