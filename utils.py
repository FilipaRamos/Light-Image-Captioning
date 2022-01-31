import os
import pickle
import configparser
import numpy as np

import transformers_tf.model as model
import transformers_tf.layers as layers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load a config file
def load_cfg(file):
    config = configparser.ConfigParser()
    config.read(file)
    return config

# Load a file
def load_file(file):
    file = open(file, 'r')
    text = file.read()
    file.close()
    return text

def load_set(file):
    doc = load_file(file)
    data = list()

    counter = 0
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        if counter == 1000:
            break
        img_id = line.split('.')[0]
        data.append(img_id)
        counter += 1
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
            desc = '<start> ' + ' '.join(img_desc) + ' <end>'
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

def array_to_str(arr):
    sentence = ' '
    for word in arr:
        sentence += word + ' '
    return sentence

def set_tokenizer(data):
    l = to_list(data)
    tokenizer = Tokenizer(num_words=5000,
                          oov_token="<unk>",
                          filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(l)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    return tokenizer

def save_tokenizer(tokenizer, file):
    with open(file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def max_length(data):
    lines = to_list(data)
    return max(len(d.split()) for d in lines)

def create_all_seqs(tokenizer, max_length, data, images, vocab_size):
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

def create_tensor_seqs(tokenizer, max_length, data, images, vocab_size):
    ''' Training Sequences
        Input:                                          Label:
        1. img startseq                                 1. previous seq + bird    
        2. img startseq, bird                           2. previous seq + flying
        3. img startseq, bird, flying                   3. previous seq + at
        4. img startseq, bird, flying, at               4. previous seq + sea
        5. img startseq, bird, flying, at, sea          5. previous seq + endseq
        6. img startseq, bird, flying, at, sea, endseq  6. -
        Shape of a sample:
        (batch_size, max_length, vocab_size)
        -1 since the last 
         '''
    x1, x2, y = list(), list(), list()
    for key, desc_l in data.items():
        for desc in desc_l:
            print('ORIGINAl DESC>', desc)
            # Encode with tokenizer
            seq = tokenizer.texts_to_sequences([desc])[0]

            for i in range(1, len(seq)):
                # Shape label
                #desc_label = np.zeros((max_length, vocab_size))

                in_seq, out_seq = seq[:i], seq[:i+1]
                print('Start>{}, End>{}'.format(in_seq, out_seq))
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                print('Padded In Seq>{}, Size>{}'.format(in_seq, len(in_seq)))

                #out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                #desc_label[-1, :] = out_seq
                #assert np.sum(desc_label[-1, :]) == 1
                out_seq = pad_sequences([out_seq], maxlen=max_length, padding='post')[0]
                print('Padded Out Seq>{}, Size>{}'.format(out_seq, len(out_seq)))

                x1.append(images[key][0])
                x2.append(in_seq)
                y.append(out_seq)
    return np.array(x1), np.array(x2), np.array(y)

def create_tensor_seqs_simple(tokenizer, max_length, data, images):
    x1, x2, y = list(), list(), list()
    for key, desc_l in data.items():
        for desc in desc_l:
            seq = tokenizer.texts_to_sequences([desc])
            seq = pad_sequences(seq, maxlen=max_length, padding='post')[0]
            #print('Padded In Seq>{}, Size>{}'.format(seq, len(seq)))

            x1.append(images[key][0])
            x2.append(seq)
    return np.array(x1), np.array(x2)

def create_seq(tokenizer, max_length, desc_l, image, vocab_size):
    x1, x2, y = list(), list(), list()
    for desc in desc_l:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out = to_categorical([out_seq], num_classes=vocab_size)[0]
            x1.append(image)
            x2.append(in_seq)
            y.append(out)
    return np.array(x1), np.array(x2), np.array(y)

def data_generator(descs, images, tokenizer, max_length, vocab_size):
    while 1:
        for key, desc_l in descs.items():
            img = images[key][0]
            in_img, in_seq, out = create_seq(tokenizer, max_length, desc_l, img, vocab_size)
            yield [in_img, in_seq], out

def word_by_id(id, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == id:
            return word
    return None

def generate_desc(model, tokenizer, img, max_length):
    input = 'startseq'
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([input])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        pred = model.predict([img, seq], verbose=0)
        pred = np.argmax(pred)
        word = word_by_id(pred, tokenizer)

        if word is None:
            break
        input += ' ' + word
        if word == 'endseq':
            break
    return input

def generate_transformer_desc(model, tokenizer, img, max_length):
    input = 'startseq'
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([input])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        # Preds will be (1, max_length, vocab_size)
        pred = model.predict([img, seq], verbose=0)
        pred = pred[:, -1:, :]
        pred = np.argmax(pred)
        word = word_by_id(pred, tokenizer)

        if word is None:
            break
        input += ' ' + word
        if word == 'endseq':
            break
    return input

def generate_transformer2d_desc(md, tokenizer, img, max_length):
    import tensorflow as tf
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0) #tokens
    result = [] #word list

    for i in range(100):
        pad_mask, look_mask = model.create_masks(output)
        comb_mask = tf.maximum(pad_mask, look_mask)

        predictions, attention_weights = md(img, output, False, comb_mask)
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        if predicted_id == end_token:
            return result,tf.squeeze(output, axis=0), attention_weights
        
        result.append(tokenizer.index_word[int(predicted_id)])
        output = tf.concat([output, predicted_id], axis=-1)

    return result, tf.squeeze(output, axis=0), attention_weights

def load_model(config):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_dir, 'config/' + config + '.cfg')
    cfg = load_cfg(cfg_file)['default']
    file_model = os.path.join(cur_dir, 'checkpoints/' + cfg['model'] + '.png')
    
    vocab_size = int(cfg['vocab_size'])
    max_length = int(cfg['max_len'])
    if cfg['model'] == 'simple':
        f_shape = int(cfg['f_shape'])
        caption_model = model.simple_caption_model(f_shape, vocab_size, max_length, file_model)
    elif cfg['model'] == 'transformer':
        f_shape = int(cfg['f_shape'])
        caption_model = model.transformer_caption_model(f_shape, vocab_size, max_length, file_model)
    elif cfg['model'] == 'transformer2d':
        caption_model = layers_tf.TransformerWrapper(
            int(cfg['NUM_LAYERS']), 
            int(cfg['EMBED_DIM']), 
            int(cfg['NUM_HEADS']), 
            int(cfg['DFF']), 
            int(cfg['ROW']), 
            int(cfg['COL']), 
            vocab_size, 
            max_length
        )
    return caption_model

def prepare(cfg):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_train = os.path.join(cur_dir, 'data/Flickr_8k.trainImages.txt')
    file_test = os.path.join(cur_dir, 'data/Flickr_8k.devImages.txt')
    file_d = os.path.join(cur_dir, 'data/descriptions.txt')
    file_tk = os.path.join(cur_dir, 'data/tokenizer.pkl')

    if any(char.isdigit() for char in cfg['model']):
        file_img = os.path.join(cur_dir, 'data/features_' + cfg['backbone'] + '2d.pkl')
    else:
        file_img = os.path.join(cur_dir, 'data/features_' + cfg['backbone'] + '.pkl')

    # Train
    data = load_set(file_train)
    print('Train dataset>%d' % len(data))
    descriptions = load_clean_descriptions(file_d, data)
    print("Train descriptions>%d" % len(descriptions))
    features = load_img_features(file_img, data)
    print("Train features size>%d" % len(features))

    tokenizer = set_tokenizer(descriptions)
    save_tokenizer(tokenizer, file_tk)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocab size>%d' % vocab_size)
    maxlen = max_length(descriptions)
    print('Max length>%d' % maxlen)

    # Test
    data_test = load_set(file_test)
    print('Test dataset>%d' % len(data_test))
    descriptions_test = load_clean_descriptions(file_d, data_test)
    print("Test descriptions>%d" % len(descriptions_test))
    features_test = load_img_features(file_img, data_test)
    print("Test features size>%d" % len(features_test))

    return descriptions, features, descriptions_test, features_test, tokenizer, vocab_size, maxlen

def check_label_correctness():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_train = os.path.join(cur_dir, 'data/Flickr_8k.trainImages.txt')
    file_test = os.path.join(cur_dir, 'data/Flickr_8k.devImages.txt')
    file_d = os.path.join(cur_dir, 'data/descriptions.txt')
    file_img = os.path.join(cur_dir, 'data/features_inception2d.pkl')
    file_tk = os.path.join(cur_dir, 'data/tokenizer.pkl')

    # Train
    data = load_set(file_train)
    print('Train dataset>%d' % len(data))
    descriptions = load_clean_descriptions(file_d, data)
    print("Train descriptions>%d" % len(descriptions))
    features = load_img_features(file_img, data)
    print("Train features size>%d" % len(features))

    tokenizer = set_tokenizer(descriptions)
    save_tokenizer(tokenizer, file_tk)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocab size>%d' % vocab_size)
    maxlen = max_length(descriptions)
    print('Max length>%d' % maxlen)

    #create_tensor_seqs(tokenizer, maxlen, descriptions, features, vocab_size)

if __name__ == "__main__":
    check_label_correctness()