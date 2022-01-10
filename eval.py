import os
import sys

import utils
import model

from nltk.translate.bleu_score import corpus_bleu

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

def eval_model(model, descs, images, tokenizer, max_length):
    realset, predset = list(), list()
    for key, desc_l in descs.items():
        pred = generate_desc(model, tokenizer, images[key], max_length)
        references = [d.split() for d in desc_l]
        realset.append(references)
        predset.append(pred.split())
    
    print('BLEU-1: %f' % corpus_bleu(realset, predset, weights=(1.0, 0, 0, 0)))
    print('BLEU-1: %f' % corpus_bleu(realset, predset, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-1: %f' % corpus_bleu(realset, predset, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-1: %f' % corpus_bleu(realset, predset, weights=(0.25, 0.25, 0.25, 0.25)))
    # Save?

def eval_all_checkpoints(checkpoints_dir, descriptions_test, features_test, tokenizer, max_length):
    for file in os.listdir(checkpoints_dir):
        if file.split('.')[1] == 'h5':
            checkpoint = os.path.join(checkpoints_dir, file)
            caption_model = load_model(checkpoint)
            eval_model(caption_model, descriptions_test, features_test, tokenizer, max_length)

def eval_checkpoint(checkpoints_dir, filename, descriptions_test, features_test, tokenizer, max_length):
    checkpoint = os.path.join(checkpoints_dir, filename)
        
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