import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from metrics.cider.cider import Cider
from nltk.translate.bleu_score import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score(references, hypothesis):
    scores = {
        'bleu-1': 0,
        'bleu-2': 0,
        'bleu-3': 0,
        'bleu-4': 0,
        'cider': 0
    }

    bleu1 = corpus_bleu(references, hypothesis, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypothesis, weights=(0.5, 0.5, 0.0, 0.0))
    bleu3 = corpus_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33, 0.0))
    bleu4 = corpus_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(
        'BLEU-1: {} <|> BLEU-2: {} <|> BLEU-3 {} <|> BLEU-4 {}'.format(
        bleu1, bleu2, bleu3, bleu4
        )
    )

    scores['bleu-1'] = bleu1
    scores['bleu-2'] = bleu2
    scores['bleu-3'] = bleu3
    scores['bleu-4'] = bleu4

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypothesis]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score_i, scores_i = Cider().compute_score(ref, hypo)
    scores['cider'] = score_i
    print('CIDEr: %f' % score_i)

    return scores

def eval_epoch(word_map, test_gen, model, device, args):
    import random
    import numpy as np
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

    hypothesis = []
    references = []
    i = 0
    losses = []
    with torch.no_grad():
        for src, tar, tar_len, caps in test_gen:
            src = src.to(device)
            tar = tar.to(device)
            tar_len = tar_len.to(device)

            preds, caption, dec_lengths, sorted_idx, alphas = model(src, tar, tar_len)
            preds_ = preds.clone()

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

            # References
            #caps = caps[sorted_idx]  # because images were sorted in the decoder
            # 2 is SOS, 0 is PAD, 3 is EOS
            for j in range(caps.shape[0]):
                img_caps = caps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in [2, 0]],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, p = torch.max(preds_, dim=2)
            p = p.tolist()
            temp_preds = list()
            for j, i in enumerate(p):
                temp_preds.append(p[j][1:dec_lengths[j]])  # remove pads
            p = temp_preds
            hypothesis.extend(p)
            
            assert len(references) == len(hypothesis)
            losses.append(loss)
    return score(references, hypothesis), losses
    