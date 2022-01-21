import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import sys
sys.path.append("..")
import metrics
from nltk.translate.bleu_score import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score(references, hypothesis):
    scores = {
        'bleu': [],
        'cider': 0
    }

    bleu1 = corpus_bleu(references, hypothesis, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypothesis, weights=(0.3, 0.3, 0.3, 0))
    bleu4 = corpus_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(
        'BLEU-1: {} <|> BLEU-2: {} <|> BLEU-3 {} <|> BLEU-4 {}'.format(
        bleu1, bleu2, bleu3, bleu4
        )
    )

    scores['bleu'].append(bleu1)
    scores['bleu'].append(bleu2)
    scores['bleu'].append(bleu3)
    scores['bleu'].append(bleu4)

    score_i, scores_i = metrics.cider.cider.Cider().compute_score(references, hypothesis)
    scores['cider'] = score_i
    print('CIDEr: %f' % score_i)

    return scores

def eval_epoch(dataset, test_gen, model, device):
    import random
    import numpy as np
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

    hypothesis = []
    references = []
    with torch.no_grad():
        for src, tar, tar_len, caps in test_gen:
            src = src.to(device)
            tar = tar.to(device)
            tar_len = tar_len.to(device)

            preds, caption, dec_lengths, sorted_idx, alphas = model(src, tar, tar_len)
            preds_ = preds.clone()

            targets = caption[:, 1:]
            preds = pack_padded_sequence(preds, dec_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, dec_lengths, batch_first=True).data
            
            loss = criterion(preds, targets)

            pred = ''
            _, out = torch.max(preds_, dim=2)
            out = out.tolist()
            print('OUT>', out)
            
            for word in out:
                if word.item() == 2: # EOS
                    break
                for key, value in dataset.tokenizer.word_index.items():
                    if value == word.item() and value != 1: # SOS
                        pred = pred + key + ' '
                        break

            caps = [t[0] for t in caps]
            ref = [d.split() for d in caps]
            references.append(ref)
            hypothesis.append(pred.split())
            
            assert len(references) == len(hypothesis)

    return score(references, hypothesis)
    