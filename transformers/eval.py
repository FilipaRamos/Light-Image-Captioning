import dataset
import simple_transformer

import torch
import torch.nn as nn

from nltk.translate.bleu_score import corpus_bleu

def eval(src_len, tar_len, vocab_size, mod='simple'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mod == 'simple':
        # Model hyperparameters
        src_vocab_size = 256
        trg_vocab_size = vocab_size
        embedding_size = 512
        num_heads = 8
        num_encoder_layers = 3
        num_decoder_layers = 3
        dropout = 0.10
        max_len_s = src_len
        max_len_t = tar_len
        forward_expansion = 4
        src_pad_idx = 0

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

    criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()
    # Parameters
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 6}
    batch_size = int(params['batch_size'])

    data = dataset.Flickr8kDataset('test')
    test_gen = torch.utils.data.DataLoader(data, **params)

    preds = []
    for src, tar in test_gen:
        print(tar)
        sentence = []
        sentence.append(sent)
        sentence[0] = 'SOS ' + sentence[0] + ' EOS'
        sentence = pad(token_caption.texts_to_sequences(sentence) , length = max_len_t)
        src , tar = create_batch(img_arr, sentence, 1, 1)
        src = torch.tensor(src).long()
        tar = torch.tensor(tar).long()
        src = src.to(device)
        tar = tar.to(device)
        
        model.eval()
        output = model(src, tar)
        loss = criterion(output.view(-1, output.shape[2]), tar.reshape(-1))
        pred = ''
        val, ind = torch.max(output.view(-1, output.shape[2]), 1)
        
        for word in ind:
            if word.item() == 1: # <end>
                    break
            for key, value in data.tokenizer.word_index.items():
                if value == word.item() and value != 2: # <start>
                    pred = pred + key + ' '
                    break
        references = [d.split() for d in desc_l]
        realset.append(references)
        predset.append(pred.split())
    
    bleu1 = corpus_bleu(realset, predset, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(realset, predset, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(realset, predset, weights=(0.3, 0.3, 0.3, 0))
    bleu4 = corpus_bleu(realset, predset, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1: %f' % bleu1)
    print('BLEU-2: %f' % bleu2)
    print('BLEU-3: %f' % bleu3)
    print('BLEU-4: %f' % bleu4)
        
def eval_epoch(dataset, test_gen, model, device):
    import random
    import numpy as np
    criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

    preds = []
    targets = []
    for src, tar, caps in test_gen:
        src, tar = np.transpose(src), np.transpose(tar)
        src = torch.tensor(src).long()
        tar = torch.tensor(tar).long()
        src = src.to(device)
        tar = tar.to(device)

        model.eval()
        output = model(src, tar)
        loss = criterion(output.view(-1, output.shape[2]), tar.reshape(-1))
        pred = ''
        val, ind = torch.max(output.view(-1, output.shape[2]), 1)
        
        for word in ind:
            if word.item() == 2: # <end>
                    break
            for key, value in dataset.tokenizer.word_index.items():
                if value == word.item() and value != 1: # <start>
                    pred = pred + key + ' '
                    break

        caps = [t[0] for t in caps]
        references = [d.split() for d in caps]
        targets.append(references)
        preds.append(pred.split())
    
    bleu1 = corpus_bleu(targets, preds, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(targets, preds, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(targets, preds, weights=(0.3, 0.3, 0.3, 0))
    bleu4 = corpus_bleu(targets, preds, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1: %f' % bleu1)
    print('BLEU-2: %f' % bleu2)
    print('BLEU-3: %f' % bleu3)
    print('BLEU-4: %f' % bleu4)