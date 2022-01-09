import os
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

def feature_extraction(in_dir, backbone='vgg'):
    if backbone == 'vgg':
        model = VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
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
        print("Done with>%s" % name)
    return features

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    in_dir = os.path.join(cur_dir, 'data/Flicker8k_Dataset')

    features = feature_extraction(in_dir)
    print("Finished extracting features>%s" % len(features))
    pickle.dump(features, open(os.path.join(cur_dir, 'data/features.pkl'), 'wb'))