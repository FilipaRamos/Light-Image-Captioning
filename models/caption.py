#!/usr/bin/env python3
import torch
import torch.nn as nn

from utils import utils

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8

    def forward(self, h, att_feats, p_att_feats):
        batch_size = h.size(0)
        att_size = att_feats.nume1() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = nn.functional.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)

        weight = nn.functional.softmax(dot, dim=1)
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

        return att_res