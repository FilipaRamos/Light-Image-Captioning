#!/usr/bin/env python3
import torch
import torch.nn as nn

class AttModel(nn.Module):
    def __init__(self):
        super(AttModel, self).__init__()