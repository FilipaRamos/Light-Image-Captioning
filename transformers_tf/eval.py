import os
import sys
import time

import utils
import transformers_tf.model as model
import transformers_tf.data_generator as data_gen

import numpy as np

from nltk.translate.bleu_score import corpus_bleu

import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def eval_model(model, cfg, descs, images, tokenizer, max_length):
    realset, predset = list(), list()
    for key, desc_l in descs.items():
        if 'transformer' in cfg.split('_')[2]:
            pred = utils.generate_transformer_desc(model, tokenizer, images[key], max_length)
        else:
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

def eval_checkpoint(checkpoints_dir, dir, descriptions_test, features_test, tokenizer, max_length, load_weights=True):
    checkpoint = os.path.join(checkpoints_dir, dir)
    
    if load_weights:
        file = os.path.join(checkpoint, dir + '.h5')
        caption_model = utils.load_model(dir)
        caption_model.load_weights(file)
    else:
        caption_model = load_model(checkpoint)
    eval_model(caption_model, dir, descriptions_test, features_test, tokenizer, max_length)

def eval_checkpoint2d(cfg, checkpoints_dir, dir, descriptions_test, features_test, tokenizer, max_length, vocab_size):
    checkpoint = os.path.join(checkpoints_dir, dir)
    file = os.path.join(checkpoint, dir + '.h5')
    
    caption_model = utils.load_model(dir)
    test_gen = data_gen.DataGenerator(vocab_size, max_length, dir, train=False)
    num_samples = int(test_gen.get_max_count())

    # Need to build model before loading weights
    f_tensor, seq_tensor, target = test_gen.__getitem__(0)
    pad_mask, look_mask = model.create_masks(seq_tensor, max_length)
    comb_mask = tf.maximum(pad_mask, look_mask)

    _, _ = caption_model(f_tensor, seq_tensor, False, comb_mask)
    caption_model.load_weights(file)
    # # # Built and loaded weights

    start = time.time()
    pb_i = Progbar(num_samples, stateful_metrics=[])

    realset, predset = list(), list()
    for key, desc_l in descriptions_test.items():
        pred, _ = utils.generate_transformer2d_desc(caption_model, tokenizer, features_test[key], max_length)
        
        references = [d.split() for d in desc_l]
        realset.append(references)
        predset.append(pred.split())
        pb_i.add(1, values=[])
    
    bleu1 = corpus_bleu(realset, predset, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(realset, predset, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(realset, predset, weights=(0.3, 0.3, 0.3, 0))
    bleu4 = corpus_bleu(realset, predset, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1: %f' % bleu1)
    print('BLEU-2: %f' % bleu2)
    print('BLEU-3: %f' % bleu3)
    print('BLEU-4: %f' % bleu4)
    print('Time taken for eval {} secs\n'.format(time.time() - start))
    
    return bleu1, bleu2, bleu3, bleu4

def eval(config):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_dir, 'config/' + config + '.cfg')
    cfg =  utils.load_cfg(cfg_file)['default']
    # Prepare data
    _, _, descriptions_test, features_test, tokenizer, vocab_size, max_length = utils.prepare(cfg)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints = os.path.join(cur_dir, 'checkpoints')
    
    # Depecrated
    #if sys.argv[1] == 'all':
    #    eval_all_checkpoints(checkpoints, descriptions_test, features_test, tokenizer, max_length)
    #else:
    if cfg['model'] == 'transformer2d':
        eval_checkpoint2d(cfg, checkpoints, sys.argv[1], descriptions_test, features_test, tokenizer, max_length, vocab_size)
    else:
        eval_checkpoint(checkpoints, sys.argv[1], descriptions_test, features_test, tokenizer, max_length, sys.argv[2])

if __name__ == "__main__":
    print('<Eval>')
    eval(sys.argv[1])
    print('<Done>')