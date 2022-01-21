import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import eval
import utils
import transformer
import simple_transformer

#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(name, epoch, model, optimizer, metrics, args):
    state = {'epoch': epoch,
             'metrics': metrics,
             'model': model,
             'optimizer': optimizer,
             'args': args}

    filename = './tmp/checkpoint_' + name + '.pth.tar'
    torch.save(state, filename)

# Model hyperparameters
args = {
    'src_vocab_size': 256,
    'trg_vocab_size': 7579,
    'embed_dim': 512,
    'n_heads': 8,
    'num_enc_layers': 3,
    'num_dec_layers': 3,
    'dropout': 0.10,
    'max_len_s': 2048,
    'max_len_t': 35,
    'att_method': 'pixel',
    'src_pad_idx': 0
}

# Training hyperparameters
learning_rate = 1e-4

'''model = simple_transformer.Transformer(
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
).to(device)'''
model = transformer.Transformer(
    args['embed_dim'],
    args['num_enc_layers'],
    args['num_dec_layers'],
    args['n_heads'],
    args['trg_vocab_size'],
    args['max_len_t'],
    args['att_method']
).to(device)

epochs = 10
pad_idx = 0
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).cuda()

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}
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
    for src, tar, tar_len in train_gen:        
        src = src.to(device)
        tar = tar.to(device)
        tar_len = tar_len.to(device)
        optimizer.zero_grad()

        preds, caption, dec_lengths, sorted_idx, alphas = model(src, tar, tar_len)
        
        targets = caption[:, 1:]
        preds = pack_padded_sequence(preds, dec_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, dec_lengths, batch_first=True).data
        
        loss = criterion(preds, targets)
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
    
    metrics = eval.eval_epoch(data_val, val_gen, model, device)
    torch.save('transformer_pixel', epoch, model, optimizer, metrics, args)