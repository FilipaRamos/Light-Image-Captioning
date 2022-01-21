import os
import sys
import pickle

import utils

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input

cur_dir = os.path.dirname(os.path.abspath(__file__))

def feature_extraction(in_dir, cfg):
    if cfg['backbone'] == 'vgg':
        tg_size = 224
        model = VGG19(weights='imagenet')
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    elif cfg['backbone'] == 'inception':
        tg_size = 299
        if any(char.isdigit() for char in cfg['model']):
            model = InceptionV3(weights='imagenet', include_top=False)
        else:
            model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    elif cfg['backbone'] == 'resnet':
        tg_size = 224
        model = ResNet101(weights='imagenet', include_top=False, pooling='avg')
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    print(model.summary())
    # Pre-extract features for all images in the dataset
    features = dict()
    for name in os.listdir(in_dir):
        # Load sample
        filename = in_dir + '/' + name
        img = image.load_img(filename, target_size=(tg_size, tg_size))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        # Use Keras API to prepare image for model
        img = preprocess_input(img)
        ft = model.predict(img, verbose=0)
        img_id = name.split('.')[0]
        features[img_id] = ft
        print("Done with>%s" % name)
    return features

if __name__ == "__main__":
    '''
    Feature Extraction
    @args: [config_name]
    @return: save pkl with features
    '''
    print('<Feature Extraction>')
    in_dir = os.path.join(cur_dir, 'data/Flicker8k_Dataset')

    if len(sys.argv) > 1:
        cfg_f = os.path.join(cur_dir, 'config/' + sys.argv[1] + '.cfg')
    else:
        cfg_f = os.path.join(cur_dir, 'config/flickr_inception_transformer.cfg')
    cfg = utils.load_cfg(cfg_f)['default']

    features = feature_extraction(in_dir, cfg)
    print("Finished extracting features>%s" % len(features))
    if any(char.isdigit() for char in cfg['model']):
        features_path = os.path.join(cur_dir, 'data/features_' + cfg['backbone'] + '2d.pkl')
    else:
        features_path = os.path.join(cur_dir, 'data/features_' + cfg['backbone'] + '.pkl')
    pickle.dump(features, open(features_path, 'wb'))
    print('<Done>')