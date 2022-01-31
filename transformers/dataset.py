import os
import json
import h5py
import torch
import random
import imageio
import numpy as np

from PIL import Image

class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(self, mode, features, transform=None, tokenizer=None, maxlen=None):
        self.mode = mode
    
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cur_dir = cur_dir.replace('transformers', '')

        self.file_f = None
        self.features = None
        if features is not None:
            self.file_f = os.path.join(cur_dir, 'data/' + features + '.pkl')
            self.features = self.load_img_features(self.file_f)
        self.img_path = os.path.join(cur_dir, 'data/Flicker8k_Dataset')

        if mode == 'train':
            file_t = os.path.join(cur_dir, 'data/Flickr_8k.trainImages.txt')
        else:
            file_t = os.path.join(cur_dir, 'data/Flickr_8k.devImages.txt')
        self.file_d = os.path.join(cur_dir, 'data/descriptions.txt')

        self.transform = transform
        self.ids = self.load_ids(file_t)
        caps = self.load_caps()
        
        if self.mode == 'train':
            self.tokenize(caps)
            self.maxlen = self.max_length(self.caps)
        else:
            self.tokenizer = tokenizer
            self.maxlen = maxlen
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]

        # Get Caption Candidates (remove SOS and EOS)
        caps_list = self.caps[id]

        # Choose a random caption for training
        cap = random.choice(caps_list)
        # +2 for SOS and EOS
        cap_len = len(cap.split(' ')) + 2
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
        image = torch.FloatTensor(image / 255.)
        cap = torch.LongTensor(cap)
        cap_len = torch.LongTensor([cap_len])

        if self.transform is not None:
            # Apply normalization
            image = self.transform(image)
        if self.mode == 'train':
            return image, cap, cap_len
        else:
            caps_list = [cap_.replace('SOS ', '') for cap_ in caps_list]
            caps_list = [cap_.replace(' EOS', '') for cap_ in caps_list]
            cp_list = []
            for c in caps_list:
                cp_list.append(self.prep_cap(c))
            cp_list = np.asarray(cp_list)
            cp_list = torch.LongTensor(cp_list)

            return image, cap, cap_len, cp_list

    def max_length(self, descs):
        dc = list()
        for key in descs.keys():
            [dc.append(d) for d in descs[key]]
        return max(len(d.split()) for d in dc)

    def prep_cap(self, cap):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = self.tokenizer.texts_to_sequences([cap])[0]
        return pad_sequences([seq], maxlen=self.maxlen + 2, padding='post')[0]

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

    def load_desc(self, file, ids):
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
        img = imageio.imread(path)
        # img = imread(impaths[i])
        if len(img.shape) == 2:
            # gray-scale
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)  # [256, 256, 1+1+1]
        img = np.array(Image.fromarray(img).resize((256, 256)))
        # img = imresize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 256, 256)
        return img

    def load_caps(self):
        self.caps = self.load_desc(self.file_d, self.ids)
        caps = list()
        for key in self.caps.keys():
            [caps.append(d) for d in self.caps[key]]
        return caps

    def tokenize(self, caps):
        from tensorflow.keras.preprocessing.text import Tokenizer
        self.tokenizer = Tokenizer(num_words=6000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', oov_token='<unk>')
        self.tokenizer.fit_on_texts(caps)

class COCODataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored - /Users/skye/docs/image_dataset/dataset
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation or testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size