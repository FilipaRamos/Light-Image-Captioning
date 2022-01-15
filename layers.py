from curses.ascii import FF
import numpy as np
import tensorflow as tf

import tensorflow.keras.layers as layers

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(
                                    input_dim=vocab_size, 
                                    output_dim=embed_dim
                                    )
        self.pos_emb = layers.Embedding(
                                    input_dim=maxlen, 
                                    output_dim=embed_dim
                                    )
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)

        x = self.token_emb(x)
        return x + positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class AnglePositionEmbedding():
    def __init__(self, row, col, embed_dim):
        #super(AnglePositionEmbedding, self).__init__()
        self.row = row
        self.col = col
        self.embed_dim = embed_dim

    def get_angles(self, pos, idx, dim):
        angle_rate = 1 / np.power(10000, (2 * (idx // 2) / np.float32(dim)))
        return pos * angle_rate

    def pos_encoding(self):
        assert self.embed_dim % 2 == 0
        row_ = np.repeat(np.arange(self.row), self.col)[:, np.newaxis]
        col_ = np.repeat(np.expand_dims(np.arange(self.col), 0), self.row, axis=0).reshape(-1, 1)
        
        angle_row = self.get_angles(row_, np.arange(self.embed_dim // 2)[np.newaxis, :], self.embed_dim // 2)
        angle_col = self.get_angles(col_, np.arange(self.embed_dim // 2)[np.newaxis, :], self.embed_dim // 2)

        angle_row[:, 0::2] = np.sin(angle_row[:, 0::2])
        angle_row[:, 1::2] = np.cos(angle_row[:, 1::2])
        angle_col[:, 0::2] = np.sin(angle_col[:, 0::2])
        angle_col[:, 1::2] = np.cos(angle_col[:, 1::2]) 

        pos_encoding = np.concatenate([angle_row, angle_col], axis=1)[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

class FeatureEncoder(layers.Layer):
    def __init__(self, embed_dim, rate=0.5):
        super(FeatureEncoder, self).__init__()
        self.dropout = Dropout(rate)
        self.dense = Dense(embed_dim, activation='relu')
    
    def call(self, x, training):
        x = self.dropout(x, training=training)
        return self.dense(x)

class CustomMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, embed_dim):
        super(CustomMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert self.embed_dim % self.num_heads == 0
        self.depth = self.embed_dim // self.num_heads
        
        self.q = Dense(self.embed_dim)
        self.k = Dense(self.embed_dim)
        self.v = Dense(self.embed_dim)
        self.dense = Dense(self.embed_dim)

    def scaled_prod_att(self, q, k, v, mask=None):
        # out: (batch_size, max_length(q), seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_att_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_att_logits += (mask * -1e9)

        att_weights = tf.nn.softmax(scaled_att_logits, axis=-1)
        # out: (batch_size, max_length(q), depth_v)
        out = tf.matmul(att_weights, v)
        return out, att_weights      

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        # out: (batch_size, max_length, embed_dim)
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        # out: (batch_size, num_heads, max_length(q, k, v), depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # out: (batch_size, max_length(q, k, v), num_heads, depth)
        scaled_att, att_weights = self.scaled_prod_att(q, k, v, mask)
        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])
        # out: (batch_size, max_length(q), embed_dim)
        concat_att = tf.reshape(scaled_att, (batch_size, -1, self.embed_dim))

        out = self.dense(concat_att)
        return out, att_weights

class EncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = CustomMultiHeadAttention(num_heads, embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation='relu'),  # (batch_size, max_length, dff)
            Dense(embed_dim)  # (batch_size, max_length, d_model)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask=None):
        # out: (batch_size, max_length, embed_dim)
        att_out, _ = self.mha(x, x, x, mask)
        att_out = self.dropout1(att_out, training=training)
        out1 = self.norm1(x + att_out)
        # out: (batch_size, input_seq, embed_dim)
        ffn_out = self.ffn(out1)
        fnn_out = self.dropout2(ffn_out, training=training)
        return self.norm2(out1 + fnn_out)

class EncoderLayerTPE(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(EncoderLayerTPE, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation='relu'), 
            Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training, mask=None):
        att_out = self.att(inputs, inputs, inputs, mask)
        att_out = self.dropout1(att_out, training=training)
        out1 = self.norm1(inputs + att_out)
        fnn_out = self.ffn(out1)
        fnn_out = self.dropout2(fnn_out, training=training)
        return self.norm2(out1 + fnn_out)

class DecoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = CustomMultiHeadAttention(num_heads, embed_dim)
        self.mha2 = CustomMultiHeadAttention(num_heads, embed_dim)

        self.ffn = Sequential([
            Dense(embed_dim, activation='relu'), 
            Dense(dff),
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_out, training, look_ahead_mask=None, padding_mask=None):
        # out: (batch_size, seq_len, embed_dim)
        att1, att1_weights = self.mha1(x, x, x, look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        out1 = self.norm1(att1 + x)

        # out: (batch_size, seq_len, embed_dim)
        att2, att2_weights = self.mha2(enc_out, enc_out, out1, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.norm2(att2 + out1)

        # out: (batch_size, seq_len, embed_dim)
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.norm3(ffn_out + out2)

        return out3, att1_weights, att2_weights

class DecoderLayerTPE(tf.keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, rate=0.1):
        super(DecoderLayerTPE, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.att1 = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim
        )
        self.att2 = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim
        )
        self.dense = Sequential([
            Dense(latent_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        att_out1 = self.att1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out1 = self.norm1(inputs + att_out1)

        att_out2 = self.att2(
            query=out1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask
        )
        out2 = self.norm2(out1 + att_out2)
        d_out = self.dense(out2)
        return self.norm3(out2 + d_out)
        
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_length = input_shape[0], input_shape[1]
        i = tf.range(seq_length)[:, tf.newaxis]
        j = tf.range(seq_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

class Encoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, dff, row, col, rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        self.embedding = Dense(self.embed_dim, activation='relu')
        self.pos_encoding = AnglePositionEmbedding(row, col, embed_dim).pos_encoding()

        self.enc_layers = [
            EncoderLayer(embed_dim, num_heads, dff, rate) for _ in range(self.num_layers)
        ]
        self.dropout = Dropout(rate)

    def call(self, x, training, mask=None):
        max_length = tf.shape(x)[1]
        # 2D Embedding out: (batch_size, max_length(HxW), embed_dim)
        x = self.embedding(x)
        x += self.pos_encoding[:, :max_length, :]

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        # out: (batch_size, max_length, embed_dim)
        return x

class Decoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, dff, vocab_size, max_length, rate=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.embedding = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
        
        self.dec_layers = [
            DecoderLayer(self.embed_dim, num_heads, dff, rate) for _ in range(self.num_layers)
        ]
        self.dropout = Dropout(rate)

    def call(self, x, enc_out, training, look_ahead_mask=None, padding_mask=None):
        max_length = tf.shape(x)[1]

        # out: (batch_size, max_length, embed_dim)
        x = self.embedding(x)
        x = x[:, :max_length, :]

        x = self.dropout(x, training=training)

        att_weights = {}
        for i in range(self.num_layers):
            x, att1_weights, att2_weights = self.dec_layers(x, enc_out, training, look_ahead_mask, padding_mask)

            att_weights['decoder_layer{}_att1'.format(i+1)] = att1_weights
            att_weights['decoder_layer{}_att2'.format(i+1)] = att2_weights

        return x, att_weights

class TransformerWrapper(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, dff, row, col, vocab_size, max_length, rate=0.1):
        super(TransformerWrapper, self).__init__()
        self.encoder = Encoder(num_layers, embed_dim, num_heads, dff, row, col, rate)
        self.decoder = Decoder(num_layers, embed_dim, num_layers, dff, vocab_size, max_length, rate)
        self.dense = Dense(vocab_size, activation="softmax")

    def call(self, features, descs, training, look_ahead_mask=None, dec_padding_mask=None, enc_padding_mask=None):
        # out: (batch_size, max_length, embed_dim)
        enc_out = self.encoder(features, training, enc_padding_mask)
        dec_out, att_weights = self.decoder(descs, enc_out, training, look_ahead_mask, dec_padding_mask)
        # out: (batch_size, max_length, vocab_size)
        out = self.dense(dec_out)
        return out, att_weights