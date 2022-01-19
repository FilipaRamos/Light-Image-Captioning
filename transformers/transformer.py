import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import simple_transformer

import torch
import torch.nn as nn
import torch.optim as optim

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import dataset

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir.split('/')[-1] == 'transformers':
    cur_dir = cur_dir.replace('transformers', '')
data_path = os.path.join(cur_dir, 'data-30/flickr30k_images/flickr30k_images')

metadata = pd.read_csv(os.path.join(cur_dir, 'data-30/flickr30k_images/results.csv'), delimiter='|', engine='python')
metadata = metadata.dropna()
is_NaN = metadata.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = metadata[row_has_NaN]
print('Roads with NaN>', rows_with_NaN)
metadata.head()

print('Nr unique imgs>', len(metadata['image_name'].unique()))

def load_image(name):
    img = image.load_img(name,target_size=(32,32,3))
    img = image.img_to_array(img)
    img = np.reshape(img,(32*32*3))
    return img

image_arr = []
sentence_arr = []
for ind in range(5000):
    if ind % 5 != 0:
        continue
    image_location = (metadata.iloc[ind,:]['image_name'])
    sentence = (metadata.iloc[ind,:][' comment'])
    
    image_arr.append(load_image(os.path.join(data_path, str(image_location))))
    sentence_arr.append('<SOS>' + sentence + '<EOS>')
Images =  np.array(image_arr)

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(x)
    t = tokenizer.texts_to_sequences(x)
    return t, tokenizer

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    padding = pad_sequences(x, padding='post', maxlen=length)
    return padding

def preprocess(sentences):
    text_tokenized, text_tokenizer = tokenize(sentences)
    text_pad = pad(text_tokenized)
    return text_pad, text_tokenizer

caption, token_caption = preprocess(sentence_arr)
print("Sentence vocabulary size:", len(token_caption.word_index))
print("Sentence Longest sentence size:", len(caption[0]))

print(Images.shape, caption.shape)

def create_batch(src, tar , batchsize , i):
    src, tar = np.transpose(src[(i-1) * batchsize : (i-1) * batchsize + batchsize]) , np.transpose(tar[(i-1) * batchsize : (i-1) * batchsize + batchsize])
    return torch.tensor(src).long(), torch.tensor(tar).long()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
src_vocab_size = 256
trg_vocab_size = 7577
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len_s = Images.shape[1]
max_len_t = 34
forward_expansion = 4
src_pad_idx = 0

# Training hyperparameters
learning_rate = 3e-4

model = simple_transformer.Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len_s,
    max_len_t,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

epochs = 1
pad_idx = 0
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).cuda()

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}
batch_size = int(params['batch_size'])

data = dataset.Flickr8kDataset('train')
train_gen = torch.utils.data.DataLoader(data, **params)

def train():
    model.train() # Turn on the train mode
    
    total_loss = 0
    start_time = time.time()
    i = 0
    for src, tar in train_gen:
        src, tar = np.transpose(src), np.transpose(tar)
        #src = torch.tensor(src).long()
        #tar = torch.tensor(tar).long()
        src = src.clone().detach().long()
        tar = tar.clone().detach().long()
        #src = src.clone().detach().requires_grad_(True).long()
        #tar = tar.clone().detach().requires_grad_(True).long()
        
        src = src.to(device)
        tar = tar.to(device)
        optimizer.zero_grad()


        output = model(src, tar)
        loss = criterion(output.view(-1, output.shape[2]), tar.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        cur_loss = loss.item()
        total_loss += cur_loss

        i += 1
        log_interval = 100
        if i % log_interval == 0 and i > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  's/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, (src.shape[1]) // batch_size, 
                    elapsed  / log_interval,
                    cur_loss, math.exp(cur_loss)))
            start_time = time.time()
    return total_loss

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    loss = train()
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | Training loss {:5.2f} | '
          .format(epoch, (time.time() - epoch_start_time),
                                     loss))
    torch.save(model.state_dict(), "simple_transformer.pt")

def display_image(name):
    from PIL import Image
    img = image.load_img(name,target_size=(512,512,3))
    img = image.img_to_array(img)
    p_image = Image.fromarray(np.uint8(img)).convert('RGB')
    #img = Image.open(p_image)
    #img = img/255
    p_image.save('eval.png')
    

def eval(index):
    image_location, sent = metadata.iloc[index,0], metadata.iloc[index,2]
    image_arr = []
    img = load_image(os.path.join(data_path, str(image_location)))
    image_arr.append(img)
    img_arr = np.array(image_arr)
    
    sentence = []
    sentence.append(sent)
    print('EVAL--->', sentence)
    sentence[0] = '<start> ' + sentence[0] + ' <end>'
    print('EVAL>', sentence)
    sentence = pad(token_caption.texts_to_sequences(sentence), length=max_len_t)
    src , tar = create_batch(img_arr, sentence, 1, 1)
    src = src.to(device)
    tar = tar.to(device)
    
    model.eval()
    output =  model(src, tar)
    loss = criterion(output.view(-1, output.shape[2]), tar.reshape(-1))
    pred = ''
    val, ind = torch.max(output.view(-1, output.shape[2]), 1)
    
    for word in ind:
        if word.item() == 1: # EOS
                break
        for key, value in token_caption.word_index.items():
            if value == word.item() and value != 2: # sos
                pred = pred + key + ' '
                break
    
    display_image(os.path.join(data_path, str(image_location)))
    print('Img>', image_location)
    return pred , loss

pred, loss = eval(0)
print('Predicted caption>--- {} --- with loss>{}'.format(pred, loss))

pred, loss = eval(10)
print('Predicted caption>--- {} --- with loss>{}'.format(pred, loss))

pred, loss = eval(20)
print('Predicted caption>--- {} --- with loss>{}'.format(pred, loss))