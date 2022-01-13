import numpy as np
import tensorflow as tf

import tensorflow.keras.layers as layers

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

class FeatureEncoder(layers.Layer):
    def __init__(self, embed_dim, rate=0.5):
        super(FeatureEncoder, self).__init__()
        self.dropout = Dropout(rate)
        self.dense = Dense(embed_dim, activation='relu')
    
    def call(self, x, training):
        x = self.dropout(x, training=training)
        return self.dense(x)

class Embedding2D(layers.Layer):
    def __init__(self, row_size, col_size, f_shape, max_length, rate=0.1):
        super(Embedding2D, self).__init__()
        self.max_length = max_length
        self.embedding = Dense(f_shape, activation='relu')
        self.pos_encoding = layers.Embedding(
                                    input_dim=f_shape, 
                                    output_dim=f_shape
                                    )
        self.dropout = Dropout(rate)

    def call(self, x, training, mask=None):
        x = self.embedding(x)
        x_ = self.pos_encoding(x)
        return self.dropout(x, training=training)

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

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
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

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, rate=0.1):
        super(TransformerDecoder, self).__init__()
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