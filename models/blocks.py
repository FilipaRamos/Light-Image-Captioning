#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential

from utils import utils
from model import AttModel

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_h_size = opt.att_h_size

        self.h2att = nn.Linear(self.rnn_size, self.att_h_size)
        self.alpha_net = nn.Linear(self.att_h_size, 1)
        self.min_value = -1e8

    def forward(self, h, att_feats, p_att_feats):
        batch_size = h.size(0)
        att_size = att_feats.nume1() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_h_size)

        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = nn.functional.tanh(dot)
        dot = dot.view(-1, self.att_h_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)

        weight = nn.functional.softmax(dot, dim=1)
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

        return att_res

class Attention2(nn.Module):
    def __init__(self, opt):
        super(Attention2, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_h_size = opt.att_h_size

        self.h1att = nn.Linear(self.rnn_size, self.att_h_size)
        self.alpha_net = nn.Linear(self.att_h_size, 1)
        self.min_value = -1e8

    def forward(self, h, att_feats, p_att_feats, mask):
        batch_size = h.size(0)
        att_size = att_feats.nume1() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_h_size)

        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = nn.functional.tanh(dot)
        dot = dot.view(-1, self.att_h_size)

        hAflat = self.alpha_net(dot)
        hAflat = hAflat.view(-1, att_size)
        hAflat.masked_fill_(mask, self.min_value)

        weight = nn.functional.softmax(hAflat, dim=1)
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

class adaPnt(nn.Module):
    def __init__(self, conv_size, rnn_size, att_h_size, dropout, min_value, beta):
        super(adaPnt, self).__init__()
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.att_h_size = att_h_size
        self.min_value = min_value
        self.conv_size = conv_size

        self.f_fc1 = nn.Linear(self.rnn_size, self.rnn_size)
        self.f_fc2 = nn.Linear(self.rnn_size, self.att_h_size)
        self.h_fc1 = nn.Linear(self.rnn_size, self.att_h_size)
        self.alpha_net = nn.Linear(self.att_h_size, 1)
        
        self.inplace = False
        self.beta = beta

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, mask):
        # Extract the batch size
        batch_size = h_out.size(0)
        # Extract regions of interest
        roi_num = conv_feat_embed.size(1)
        conv_feat_embed = conv_feat_embed.view(-1, roi_num, self.att_h_size)
        fake_region = nn.functional.relu(self.f_fc1(fake_region.view(-1, self.rnn_size)), inplace=self.inplace)
        fake_region_embed = self.f_fc2(fake_region)

        h_out_embed = self.h_fc1(h_out)
        img_all_embed = torch.cat([fake_region_embed.view(-1, 1, self.att_h_size), conv_feat_embed], 1)
        hA = nn.functional.tanh(img_all_embed + h_out_embed.view(-1, 1, self.att_h_size))
        hAflat = self.alpha_net(hA.view(-1, self.att_h_size))
        hAflat = hAflat.view(-1, roi_num + 1)
        hAflat.masked_fill_(mask, self.min_value)

        return hAflat

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.min_value = -1e8

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.rnn_size*2, opt.rnn_size)
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)

        self.adaPnt = adaPnt(opt.input_encoding_size, opt.rnn_size, opt.att_h_size, self.drop_prob_lm, self.min_value, opt.beta)
        self.i2h_2 = nn.Linear(opt.rnn_size*2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, att_mask, pnt_mask, state):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, conv_feats, p_conv_feats)
        att2 = self.attention2(h_att, pool_feats, p_pool_feats, att_mask[:, 1:])
        lang_lstm_input = torch.cat([att + att2, h_att], 1)

        ada_gate_point = nn.functional.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(state[0][1]))
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        
        output = nn.functional.dropout(h_lang, self.drop_prob_lm, self.training)
        fake_box = nn.functional.dropout(ada_gate_point*nn.functional.tanh(state[1][1]), self.drop_prob_lm, training=self.training)
        det_prob = self.adaPnt(output, fake_box, pool_feats, p_pool_feats, pnt_mask)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        
        return output, det_prob, state

class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_h_size = opt.att_h_size
        self.min_value = -1e8

        self.adaPnt = adaPnt(opt.input_encoding_size, opt.rnn_size, opt.att_h_size, self.drop_prob_lm, self.min_value, opt.beta)
        # Custom LSTM
        self.a2c1 = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.a2c2 = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 6 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 6 * self.rnn_size)

        self.dropout1 = nn.Dropout(self.drop_prob_lm)
        self.dropout2 = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, pool_feats, p_pool_feats, att_mask, pnt_mask, state):
        att_res1 = self.attention(state[0][-1], att_feats, p_att_feats)
        att_res2 = self.attention2(state[0][-1], pool_feats, p_pool_feats, att_mask[:, 1:])

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 4 * self.rnn_size)
        sigmoid_chunk = nn.functional.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        s_gate = sigmoid_chunk.narrow(1, self.rnn_size * 3, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 4 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c1(att_res1) + self.a2c1(att_res2)

        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size), in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * nn.functional.tanh(next_c)
        fake_box = s_gate * nn.functional.tanh(next_c)

        output = self.dropou1(next_h)
        fake_box = self.dropout2(fake_box)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        det_prob = self.adaPnt(output, fake_box, pool_feats, p_pool_feats, pnt_mask)
        
        return output, det_prob, state

class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)
        self.ccr_core = CascadeCore(opt)

class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__()
        self.num_layers = 1
        self.core = Att2in2Core(opt)
        self.ccr_core = CascadeCore(opt)

class CascadeCore(nn.Module):
    def __init__(self, opt):
        super(CascadeCore, self).__init__()
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fg_size = opt.fg_size
        
        self.bn_fc = nn.Sequential(
            nn.Linear(opt.rnn_size + opt.rnn_size, opt.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(opt.rnn_size, 2)
        )

        self.fg_fc = nn.Sequential(
            nn.Linear(opt.rnn_size + opt.rnn_size, opt.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(opt.rnn_size, 300)
        )

        self.fg_emb = nn.parameter.Parameter(opt.glove_fg)
        self.fg_emb.requires_grad=False

        self.fg_mask = nn.parameter.Parameter(opt.fg_mask)
        self.fg_mask.requires_grad=False
        
        self.min_value = -1e8
        self.beta = opt.beta

    def forward(self, fg_idx, pool_feats, rnn_outs, roi_labels, seq_batch_size, seq_cnt):
        roi_num = pool_feats.size(1)
        pool_feats = pool_feats.view(seq_batch_size, 1, roi_num, self.rnn_size) * \
                    roi_labels.view(seq_batch_size, seq_cnt, roi_num, 1)

        pool_cnt = roi_labels.sum(2)
        pool_cnt[pool_cnt == 0] = 1
        pool_feats = pool_feats.sum(2) / pool_cnt.view(seq_batch_size, seq_cnt, 1)

        pool_feats = torch.cat((rnn_outs, pool_feats), 2)
        bn_logprob = nn.functional.log_softmax(self.bn_fc(pool_feats), dim=2)

        fg_out = self.fg_fc(pool_feats)
        fg_score = torch.mm(fg_out.view(-1, 300), self.fg_emb.t()).view(seq_batch_size, -1, self.fg_size + 1)

        fg_mask = self.fg_mask[fg_idx]
        fg_score.masked_fill_(fg_mask.view_as(fg_score), self.min_value)
        fg_logprob = nn.functional.log_softmax(self.beta * fg_score, dim=2)

        return bn_logprob, fg_logprob
