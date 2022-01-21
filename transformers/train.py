import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import eval
import utils
import simple_transformer

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
src_vocab_size = 256
trg_vocab_size = 7579
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len_s = 2048
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

epochs = 10
pad_idx = 0
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).cuda()

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}
batch_size = int(params['batch_size'])

data = dataset.Flickr8kDataset('train', 'features_resnet')
train_gen = torch.utils.data.DataLoader(data, **params)
data_val = dataset.Flickr8kDataset('test', 'features_resnet')
val_gen = torch.utils.data.DataLoader(data_val, **params)

def train():
    model.train() # Turn on the train mode
    
    i = 0
    total_loss = 0
    start_time = time.time()
    for src, tar in train_gen:
        src, tar = np.transpose(src), np.transpose(tar)
        src = torch.tensor(src).long()
        tar = torch.tensor(tar).long()
        #print('SRC>', src.shape)
        #print('TAR>', tar.shape)
        
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
    
eval.eval_epoch(data_val, val_gen, model, device)