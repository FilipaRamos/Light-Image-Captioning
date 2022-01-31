import os
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import eval
import utils
import transformer
import simple_transformer

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

if sys.argv[1] == 'coco':
    # Read COCO's word map
    import json
    word_map_file = os.path.join('../data-coco/gen_data', 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Bolerplate args for COCO
    args = {
        'src_vocab_size': 256,
        'trg_vocab_size': len(word_map),
        'embed_dim': 512,
        'n_heads': 8,
        'num_enc_layers': 2,
        'num_dec_layers': 4,
        'dropout': 0.10,
        'max_len_s': 2048,
        'max_len_t': 52,
        'att_method': 'pixel',
        'src_pad_idx': 0,
        'dt': 'coco'
    }
else:
    # Bolerplate args for Flickr8k
    args = {
        'src_vocab_size': 256,
        'trg_vocab_size': 6000,
        'embed_dim': 512,
        'n_heads': 12,
        'num_enc_layers': 2,
        'num_dec_layers': 6,
        'dropout': 0.10,
        'max_len_s': 2048,
        'max_len_t': 36,
        'att_method': 'pixel',
        'src_pad_idx': 0,
        'dt': 'flickr'
    }

# Training hyperparameters
learning_rate = 1e-4

# Deprecated simple transformer later on
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

epochs = 80
pad_idx = 0
best_bleu4 = 0
ep_since_imp = 0
optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).cuda()
normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}
batch_size = int(params['batch_size'])

if args['dt'] == 'flickr':
    data = dataset.Flickr8kDataset('train', None, transform=transforms.Compose([normalization]))
    train_gen = torch.utils.data.DataLoader(data, **params)
    data_val = dataset.Flickr8kDataset('test', None, transform=transforms.Compose([normalization]), tokenizer=data.tokenizer, maxlen=data.maxlen)
    val_gen = torch.utils.data.DataLoader(data_val, **params)
elif args['dt'] == 'coco':
    data = dataset.COCODataset('../data-coco/gen_data', 'coco_5_cap_per_img_5_min_word_freq', 'TRAIN', transform=transforms.Compose([normalization]))
    train_gen = torch.utils.data.DataLoader(data, **params)
    data_val = dataset.COCODataset('../data-coco/gen_data', 'coco_5_cap_per_img_5_min_word_freq', 'VAL', transform=transforms.Compose([normalization]))
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

        preds, caption, dec_lengths, sorted_idx, alphas = model(src, tar, tar_len)
        
        #targets = caption[:, 1:]
        targets = caption
        preds = pack_padded_sequence(preds, dec_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, dec_lengths, batch_first=True).data

        loss = criterion(preds, targets)
        dec_alphas = alphas["dec_enc_atts"]
        alpha_trans_c = 1 / (args['n_heads'] * args['num_dec_layers'])
        for layer in range(args['num_dec_layers']):  # args.decoder_layers = len(dec_alphas)
            cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 52, 196]
            for h in range(args['n_heads']):
                cur_head_alpha = cur_layer_alphas[:, h, :, :]
                loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)

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
                    epoch, i, (tar.shape[1]) // batch_size, 
                    elapsed  / log_interval,
                    cur_loss, math.exp(cur_loss)))
            start_time = time.time()
    return total_loss

def plot_losses(loss_values):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    plt.plot(loss_values)
    fig.savefig('./tmp/loss.png')   # save the figure to file
    plt.close(fig)

loss_values = []
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    if ep_since_imp == 25:
            print("The model has not improved in the last 25 epochs")
            break
    if ep_since_imp > 0 and ep_since_imp % 5 == 0:
        print("\nDECAYING learning rate.")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.8

    loss = train()
    metrics, eval_losses = eval.eval_epoch(word_map, val_gen, model, device, args)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | Training loss {:5.2f} | '
          .format(epoch, (time.time() - epoch_start_time),
                                     loss))

    bleu4 = metrics['bleu-4']
    best = bleu4 > best_bleu4
    best_bleu4 = max(bleu4, best_bleu4)
    if best:
        ep_since_imp = 0
        save_checkpoint('transformer_pixel', epoch, model, optimizer, metrics, args)
    else:
        ep_since_imp += 1
        print("\nEpochs since last improvement: %d\n" % (ep_since_imp,))
    
    # Save loss plot
    loss_values.append(loss / train_gen.__len__())
    if epoch % 5 == 0:
        plot_losses(loss_values)
