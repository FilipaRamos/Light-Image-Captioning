import os
import sys

import utils
import model

import numpy as np

from nltk.translate.bleu_score import corpus_bleu

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def eval_model(model, descs, images, tokenizer, max_length):
    realset, predset = list(), list()
    for key, desc_l in descs.items():
        pred = utils.generate_desc(model, tokenizer, images[key], max_length)
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
    
    return bleu1, bleu2, bleu3, bleu4

def eval_all_checkpoints(checkpoints_dir, descriptions_test, features_test, tokenizer, max_length):
    best_bleu = [0,0,0,0]
    for file in os.listdir(checkpoints_dir):
        if file.split('.')[1] == 'h5':
            checkpoint = os.path.join(checkpoints_dir, file)
            caption_model = load_model(checkpoint)
            b1, b2, b3, b4 = eval_model(caption_model, descriptions_test, features_test, tokenizer, max_length)
            if b1 > best_bleu[0]: best_bleu[0] = b1
            if b2 > best_bleu[1]: best_bleu[1] = b2
            if b3 > best_bleu[2]: best_bleu[2] = b3
            if b4 > best_bleu[3]: best_bleu[3] = b4
    print("Best BLEU>")
    print("b1=%d" % best_bleu[0])
    print("b2=%d" % best_bleu[1])
    print("b3=%d" % best_bleu[2])
    print("b4=%d" % best_bleu[3])

def eval_checkpoint(checkpoints_dir, dir, descriptions_test, features_test, tokenizer, max_length):
    checkpoint = os.path.join(checkpoints_dir, dir)
        
    caption_model = load_model(checkpoint)
    eval_model(caption_model, descriptions_test, features_test, tokenizer, max_length)

def eval():
    # Prepare data
    _, _, descriptions_test, features_test, tokenizer, vocab_size, max_length = utils.prepare()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints = os.path.join(cur_dir, 'checkpoints')
    
    if sys.argv[1] == 'all':
        eval_all_checkpoints(checkpoints, descriptions_test, features_test, tokenizer, max_length)
    else:
        eval_checkpoint(checkpoints, sys.argv[1], descriptions_test, features_test, tokenizer, max_length)

if __name__ == "__main__":
    print('<Eval>')
    eval()
    print('<Done>')