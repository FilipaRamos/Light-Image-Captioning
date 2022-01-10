import os
import pickle
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load a file
def load_file(file):
    file = open(file, 'r')
    text = file.read()
    file.close()
    return text

def load_set(file):
    doc = load_file(file)
    data = list()

    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        img_id = line.split('.')[0]
        data.append(img_id)
    return set(data)

def load_clean_descriptions(file, data):
    doc = load_file(file)

    descs = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        img_id, img_desc = tokens[0], tokens[1:]
        if img_id in data:
            if img_id not in descs:
                descs[img_id] = list()
            desc = 'startseq ' + ' '.join(img_desc) + ' endseq'
            descs[img_id].append(desc)
    return descs

def load_img_features(file, data):
    features = pickle.load(open(file, 'rb'))
    features = {k: features[k] for k in data}
    return features

def to_list(data):
    descs = list()
    for key in data.keys():
        [descs.append(d) for d in data[key]]
    return descs

def set_tokenizer(data):
    l = to_list(data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(l)
    return tokenizer

def max_length(data):
    lines = to_list(data)
    return max(len(d.split()) for d in lines)

def create_seqs(tokenizer, max_length, data, images, vocab_size):
    ''' Training Sequences
        Input:
        1. img startseq
        2. img startseq, bird
        3. img startseq, bird, flying
        4. img startseq, bird, flying, at
        5. img startseq, bird, flying, at, sea
        6. img startseq, bird, flying, at, sea, endseq
         '''
    x1, x2, y = list(), list(), list()
    for key, desc_l in data.items():
        for desc in desc_l:
            # encode with tokenizer
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split into pairs
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                x1.append(images[key][0])
                x2.append(in_seq)
                y.append(out_seq)
    return np.array(x1), np.array(x2), np.array(y)

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(cur_dir, 'data/Flickr_8k.trainImages.txt')
    file_d = os.path.join(cur_dir, 'data/descriptions.txt')
    file_img = os.path.join(cur_dir, 'data/features.pkl')

    data = load_set(file)
    print('Train dataset>%d' % len(data))
    descriptions = load_clean_descriptions(file_d, data)
    print("Train descriptions>%s" % len(descriptions))
    features = load_img_features(file_img, data)
    print("Train features size>%s" % len(features))

    tokenizer = set_tokenizer(descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocab size>%d' % vocab_size)
    maxlen = max_length(descriptions)
    x1, x2, y = create_seqs(tokenizer, maxlen, descriptions, features, vocab_size)