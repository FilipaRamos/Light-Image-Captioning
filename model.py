import layers
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Add

from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

EMBED_DIM = 256
NUM_HEADS = 8
LATENT_DIM = 512

def simple_caption_model(f_shape, vocab_size, max_length, file):
    # Feature Extraction
    input1 = Input(shape=f_shape)
    f1 = Dropout(0.5)(input1)
    f2 = Dense(EMBED_DIM, activation='relu')(f1)
    # Sequence Processor
    input2 = Input(shape=(max_length,))
    s1 = Embedding(vocab_size, EMBED_DIM, mask_zero=True)(input2)
    s2 = Dropout(0.5)(s1)
    s3 = LSTM(256)(s2)
    # Decoder
    # Merge features from image and text
    d1 = Add()([f2, s3])
    d2 = Dense(EMBED_DIM, activation='relu')(d1)
    out = Dense(vocab_size, activation='softmax')(d2)
    # Model Compilation
    model = Model(inputs=[input1, input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Summary
    print(model.summary())
    plot_model(model, to_file=file, show_shapes=True)
    
    return model

def transformer_caption_model(f_shape, vocab_size, max_length, file):
    # Image Feature Extraction
    f_input = Input(shape=f_shape)
    f_out = layers.FeatureEncoder(EMBED_DIM)(f_input)
    
    # Encoder
    # enc_input (embed_dim,)
    enc_emb = layers.TokenAndPositionEmbedding(max_length, vocab_size, EMBED_DIM)(f_out)
    # Limit shape of tensor to max seq length
    enc_emb = enc_emb[:, :max_length, :]
    enc_out = layers.TransformerEncoder(EMBED_DIM, NUM_HEADS, LATENT_DIM)(enc_emb)
    # enc_out (batch_size, max_length, vocab_size)

    # Language Decoder
    dec_input = Input(shape=(max_length,), dtype="int32", name="decoder_input")
    # enc_seq_input (batch_size, max_length, vocab_size)
    x = layers.TokenAndPositionEmbedding(max_length, vocab_size, EMBED_DIM)(dec_input)
    x = layers.TransformerDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x, enc_out)
    x = Dropout(0.5)(x)
    dec_out = Dense(vocab_size, activation="softmax")(x)

    # Model Compilation
    # dec_out (batch_size, max_length, vocab_size)
    model = Model(inputs=[f_input, dec_input], outputs=dec_out, name="transformer_caption_model")
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Summary
    print(model.summary())
    plot_model(model, to_file=file, show_shapes=True)

    return model