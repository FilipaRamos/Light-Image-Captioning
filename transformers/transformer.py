from ast import Mult
from re import sub
from xml.sax.xmlreader import InputSource
import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(nn.Module):
    def __init__(self, qkv_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.qkv_dim = qkv_dim

    def forward(self, q, k, v, att_mask):
        ''' len_k = len_v
        q: (batch_size, n_heads, -1 * (len_q), qkv_dim)
        k: (batch_size, n_heads, -1 * (len_k), qkv_dim)
        v: (batch_size, n_heads, -1 * (len_v), qkv_dim)
        att_mask: (batch_size, n_heads, len_q, len_k)
        '''
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.qkv_dim)
        # Fill elements where mask is 1
        scores.to(device).masked_fill_(att_mask, -1e9)
        att = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(att, v).to(device)
        # (batch_size, n_heads, len_q, qkv_dim)
        return context, att

class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, qkv_dim, n_heads=3, rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.w_q = nn.Linear(q_dim, qkv_dim * n_heads).to(device)
        self.w_k = nn.Linear(k_dim, qkv_dim * n_heads).to(device)
        self.w_v = nn.Linear(k_dim, qkv_dim * n_heads).to(device)

        self.n_heads = n_heads
        self.qkv_dim = qkv_dim
        self.embed_dim = q_dim
        self.dropout = nn.Dropout(p=rate)
        self.w_o = nn.Linear(self.n_heads * self.qkv_dim, self.embed_dim).to(device)

    def forward(self, q, k, v, att_mask):
        """
        Self-encoder attention:
                Q = K = V: (batch_size, num_pixels=196, encoder_dim=2048)
                attn_mask: (batch_size, len_q=196, len_k=196)
        Self-decoder attention:
                Q = K = V: (batch_size, max_len=52, embed_dim=512)
                attn_mask: (batch_size, len_q=52, len_k=52)
        Encoder-decoder attention:
                Q: (batch_size, 52, 512) from decoder
                K, V: (batch_size, 196, 2048) from encoder
                attn_mask: (batch_size, len_q=52, len_k=196)
        return _, attn: (batch_size, n_heads, len_q, len_k)
        """
        residual, batch_size = q, q.size(0)
        q_s = self.w_q(q).view(batch_size, -1, self.n_heads, self.qkv_dim).transpose(1, 2)
        k_s = self.w_k(k).view(batch_size, -1, self.n_heads, self.qkv_dim).transpose(1, 2)
        v_s = self.w_v(v).view(batch_size, -1, self.n_heads, self.qkv_dim).transpose(1, 2)

        att_mask = att_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, att = ScaledDotProductAttention(self.qkv_dim)(q_s, k_s, v_s, att_mask)
        context = context.transpose(1, 2).contigous().view(batch_size, -1, self.n_heads * self.qkv_dim).to(device)
        # out: (batch_size, len_q, embed_dim)
        out = self.w_o(context)
        out = self.dropout(out)
        return nn.LayerNorm(self.embed_dim).to(device)(out + residual), att

class PointWiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff, rate):
        self.conv1 = nn.Conv1d(in_channeld=embed_dim, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channeld=d_ff, out_channels=embed_dim, kernel_size=1).to(device)
        self.dropout = nn.Dropout(p=rate)
        self.embed_dim = embed_dim
    
    def forward(self, input):
        '''
        Encoder in: (batch_size, len_q=196, embed_dim=2048)
        Decoder in: (batch_size, max_len, embed_dim=512)
        '''
        residual = input
        out = nn.ReLU()(self.conv1(input.transpose(1, 2)))
        out = self.conv2(out).transpose(1, 2)
        out = self.dropout(out)
        return nn.LayerNorm(self.embed_dim).to(device)(out + residual)

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, att_method, rate):
        super(EncoderLayer, self).__init__()
        if att_method == 'pixel':
            self.enc_att = MultiHeadAttention(q_dim=2048, k_dim=2048, qkv_dim=64, n_heads=n_heads, rate=rate)
            self.ffn = PointWiseFeedForwardNet(embed_dim=2048, d_ff=4096, rate=rate)
        elif att_method == 'channel':
            self.enc_att = MultiHeadAttention(q_dim=196, k_dim=196, qkv_dim=64, n_heads=n_heads, rate=rate)
            self.ffn = PointWiseFeedForwardNet(embed_dim=196, d_ff=512, rate=rate)

    def forward(self, input, att_mask):
        '''
        in: (batch_size, num_pixels=196, 2048)
        out: (batch_size, len_q, d_model=2048)
        att: (batch_size, n_heads, 196, 196)
        '''
        out, att = self.enc_att(input, input, input, att_mask)
        out = self.ffn(out)
        return out, att

class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads, att_method, rate):
        super(Encoder, self).__init__()
        if att_method == 'pixel':
            self.pos_emb = nn.Embedding.from_pretrained(self.get_pos_embedding_table(), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, att_method, rate) for _ in range(n_layers)])
        self.att_method = att_method

    def get_pos_embedding_table(self):
        def cal_angle(pos, h_index):
            x = pos % 14
            y = pos // 14
            x_enc = x / np.power(10000, h_index / 1024)
            y_enc = y / np.power(10000, h_index / 1024)
            return np.sin(x_enc), np.sin(y_enc)
        def get_pos_angle_vec(pos):
            return [cal_angle(pos, h_index)[0] for h_index in range(1024)] \
                + [cal_angle(pos, h_index)[1] for h_index in range(1024)]  

        embedding_table = np.array([get_pos_angle_vec(pos_i) for pos_i in range(196)])
        return torch.FloatTensor(embedding_table).to(device)              

    def forward(self, input):
        '''
        input: (batch_size, num_pixels=196, d_model=2048)
        '''
        batch_size = input.size(0)
        pos = input.size(1)
        if self.att_method == 'pixel':
            out = input + self.pos_emb(torch.LongTensor([list(range(pos))] * batch_size).to(device))
        att_mask = (torch.tensor(np.zeros((batch_size, pos, pos))).to(device) == torch.tensor(np.ones((batch_size, pos, pos))).to(device))
        enc_att = []
        for layer in self.layers:
            out, att = layer(out, att_mask)
            enc_att.append(att)
        return out, enc_att

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, att_method, rate):
        super(DecoderLayer, self).__init__()
        self.dec_att = MultiHeadAttention(q_dim=embed_dim, k_dim=embed_dim, qkv_dim=64, n_heads=n_heads, rate=rate)
        if att_method == 'pixel':
            self.dec_enc_att = MultiHeadAttention(q_dim=embed_dim, k_dim=2048, qkv_dim=64, n_heads=n_heads, rate=rate)
            self.ffn = PointWiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, rate=rate)
        elif att_method == 'channel':
            self.dec_enc_att = MultiHeadAttention(q_dim=embed_dim, k_dim=196, qkv_dim=64, n_heads=n_heads, rate=rate)
            self.ffn = PointWiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, rate=rate)
        
    def forward(self, dec_input, enc_output, dec_self_att_mask, dec_enc_att_mask):
        '''
        dec_input: (batch_size, max_len=52, embed_dim=512)
        enc_output: (batch_size, num_pixels=196, 2048)
        dec_self_attn_mask: (batch_size, 52, 52)
        dec_enc_attn_mask: (batch_size, 52, 196)
        '''
        out, self_att = self.dec_att(dec_input, dec_input, dec_input, dec_self_att_mask)
        out, dec_enc_att = self.dec_enc_att(out, enc_output, enc_output, dec_enc_att_mask)
        out = self.ffn(out)
        return out, self_att, dec_enc_att

class Decoder(nn.Module):
    def __init__(self, n_layers, n_heads, embed_dim, att_method, vocab_size, maxlen, rate):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen = maxlen

        self.tg_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(self.get_pos_embedding_table(embed_dim), freeze=True)
        self.dropout = nn.Dropout(p=rate)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, n_heads, att_method, rate) for _ in range(n_layers)])
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
        self.att_method = att_method

    def get_pos_embedding_table(self, embed_dim):
        def cal_angle(pos, h_index):
            return pos / np.power(10000, 2 * (h_index // 2) / embed_dim)
        def get_pos_angle_vec(pos):
            return [cal_angle(pos, h_index) for h_index in range(embed_dim)]

        embedding_table = np.array([get_pos_angle_vec(pos_i) for pos_i in range(52)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2]) # 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2]) # 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_att_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()

        # out: (batch_size, 1, len_k)
        pad_mask = seq_k.data.eq(0).unsqueeze(1)
        # out: (batch_size, len_q, len_k)
        return pad_mask.expand(batch_size, len_q, len_k)

    def get_att_sub_mask(self, seq):
        att_shape = [seq.size(0), seq.size(1), seq.size(1)]
        sub_mask = np.triu(np.ones(att_shape), k=1)
        sub_mask = torch.from_numpy(sub_mask).byte().to(device)
        return sub_mask

    def forward(self, enc_out, captions, lengths):
        '''
        enc_out: (batch_size, num_pixels=196, 2048)
        captions: (batch_size, 52)
        lengths: (batch_size, 1)
        '''
        batch_size = enc_out.size(0)

        '''
        out: (batch_size, max_len, embed_dim)
        pad_mask: (batch_size, len_q, len_k), 1 if pad=0
        sub_mask: (batch_size, maxlen, maxlen)
        dec_att_mask: val > 0 masked
        dec_enc_att_mask: enc-dec mask
        '''
        dec_out = self.tg_emb(captions) + self.pos_emb(torch.LongTensor([list(range(self.maxlen))] * batch_size).to(device))
        dec_out = self.dropout(dec_out)
        pad_mask = self.get_att_pad_mask(captions, captions)
        sub_mask = self.get_att_sub_mask(captions)
        att_mask = torch.gt((pad_mask + sub_mask), 0)
        if self.att_method == 'pixel':
            dec_enc_mask = (torch.tensor(np.zeros((batch_size, self.maxlen, 196))).to(device) == torch.tensor(np.ones((batch_size, self.maxlen, 196))).to(device))
        elif self.att_method == 'channel':
            dec_enc_mask = (torch.tensor(np.zeros((batch_size, self.maxlen, 512))).to(device) == torch.tensor(np.ones((batch_size, self.maxlen, 512))).to(device))

        dec_atts, dec_enc_atts = [], []
        for layer in self.layers:
            dec_out, dec_att, dec_enc_att = layer(dec_out, enc_out, att_mask, dec_enc_mask)
            dec_atts.append(dec_att)
            dec_enc_atts.append(dec_enc_att)
        preds = self.projection(dec_out)
        return preds, captions, dec_atts, dec_enc_atts

class Transformer(nn.Module):
    def __init__(self, embed_dim, enc_layers, dec_layers, n_heads=8, att_method='pixel', vocab_size, maxlen, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_layers, n_heads, att_method, rate)
        self.decoder = Decoder(dec_layers, n_heads, embed_dim, att_method, vocab_size, maxlen, rate)
        self.embedding = self.decoder.tg_emb
        self.att_method = att_method
    
    def load_pretrained_embeddings(self, emb):
        self.embedding.weight = nn.Parameter(emb)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, enc_input, caption):
        '''
        
        '''
        batch_size = enc_input.size(0)
        enc_dim = enc_input.size(-1)
        if self.att_method == 'pixel':
            enc_input = enc_input.view(batch_size, -1, enc_dim)
        elif self.att_method == 'channel':
            enc_input = enc_input.view(batch_size, -1, enc_dim).permute(0, 2, 1) # (batch_size, 2048, 196)
        
        