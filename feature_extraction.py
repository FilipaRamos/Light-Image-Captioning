import os
import sys
import pickle

import utils

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

cur_dir = os.path.dirname(os.path.abspath(__file__))

def feature_extraction(in_dir, cfg):
    if cfg['default']['backbone'] == 'vgg':
        model = VGG19(weights='imagenet')
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    elif cfg['default']['backbone'] == 'inception':
        model = InceptionV3(weights='imagenet', include_top=False)
        #print(model.layers)
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    # Pre-extract features for all images in the dataset
    features = dict()
    for name in os.listdir(in_dir):
        # Load sample
        filename = in_dir + '/' + name
        img = image.load_img(filename, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        # Use Keras API to prepare image for model
        img = preprocess_input(img)
        ft = model.predict(img, verbose=0)
        img_id = name.split('.')[0]
        features[img_id] = ft
        print(ft.shape)
        print("Done with>%s" % name)
    return features

if __name__ == "__main__":
    in_dir = os.path.join(cur_dir, 'data/Flicker8k_Dataset')

    if len(sys.argv) > 1:
        cfg_f = os.path.join(cur_dir, 'config/' + sys.argv[1])
    else:
        cfg_f = os.path.join(cur_dir, 'config/flickr_inception_simple.cfg')
    cfg = utils.load_cfg(cfg_f)

    features = feature_extraction(in_dir, cfg)
    print("Finished extracting features>%s" % len(features))
    pickle.dump(features, open(os.path.join(cur_dir, 'data/features_' + cfg['default']['backbone'] + '.pkl'), 'wb'))