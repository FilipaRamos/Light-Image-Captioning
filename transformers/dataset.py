import os
import torch
import random
import numpy as np

class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(self, mode, features):
        self.mode = mode
    
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cur_dir = cur_dir.replace('transformers', '')

        self.file_f = None
        if features is not None:
            self.file_f = os.path.join(cur_dir, 'data/' + features + '.pkl')
        self.img_path = os.path.join(cur_dir, 'data/Flicker8k_Dataset')

        if mode == 'train':
            file_t = os.path.join(cur_dir, 'data/Flickr_8k.trainImages.txt')
        else:
            file_t = os.path.join(cur_dir, 'data/Flickr_8k.devImages.txt')
        self.file_d = os.path.join(cur_dir, 'data/descriptions.txt')
        
        self.ids = self.load_ids(file_t)
        self.features = self.load_img_features(self.file_f)
        
        self.tokenize()
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.maxlen = self.max_length(self.caps)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]

        # Get Caption Candidates (remove SOS and EOS)
        caps_list = self.caps[id]

        # Choose a random caption for training
        cap = random.choice(caps_list)
        cap_len = len(cap.split(' '))
        # The caption has at least SOS, EOS and a word
        assert cap_len >= 3
        cap = self.prep_cap(cap)

        # Get Img
        if self.features is not None:
            image = self.features[id][0]
        else:
            img = os.path.join(self.img_path, id + '.jpg')
            image = self.load_image(img)

        # Convert to tensors
        image = torch.FloatTensor(image)
        cap = torch.LongTensor(cap)
        cap_len = torch.LongTensor([cap_len])

        if self.mode == 'train':
            return image, cap, cap_len
        else:
            caps_list = [cap.replace('SOS ', '') for cap in caps_list]
            caps_list = [cap.replace(' EOS', '') for cap in caps_list]
            return image, cap, cap_len, caps_list

    def max_length(self, descs):
        dc = list()
        for key in descs.keys():
            [dc.append(d) for d in descs[key]]
        return max(len(d.split()) for d in dc)

    def prep_cap(self, cap):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = self.tokenizer.texts_to_sequences([cap])[0]
        return pad_sequences([seq], maxlen=self.maxlen + 1, padding='post')[0]

    def load_ids(self, file):
        ids = []
        with open(file) as file:
            for line in file:
                id = line.rstrip()
                ids.append(id.split('.')[0])
        return ids

    def load_img_features(self, file):
        import pickle
        features = pickle.load(open(file, 'rb'))
        return features

    def load_descs(self, file, ids):
        file = open(file, 'r')
        text = file.read()
        file.close()

        descs = dict()
        for line in text.split('\n'):
            tokens = line.split()
            img_id, img_desc = tokens[0], tokens[1:]
            if img_id in ids:
                if img_id not in descs:
                    descs[img_id] = list()
                desc = 'SOS ' + ' '.join(img_desc) + ' EOS'
                descs[img_id].append(desc)
        return descs

    def load_image(self, path):
        from tensorflow.keras.preprocessing import image
        img = image.load_img(path, target_size=(32, 32, 3))
        img = image.img_to_array(img)
        img = np.reshape(img, (32*32*3))
        return img

    def tokenize(self):
        from tensorflow.keras.preprocessing.text import Tokenizer
        self.caps = self.load_descs(self.file_d, self.ids)
        caps = list()
        for key in self.caps.keys():
            [caps.append(d) for d in self.caps[key]]

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(caps)
