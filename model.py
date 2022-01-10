import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Add

from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

def simple_caption_model(vocab_size, max_length, file):
    # Feature Extraction
    input1 = Input(shape=(4096,))
    f1 = Dropout(0.5)(input1)
    f2 = Dense(256, activation='relu')(f1)
    # Sequence Processor
    input2 = Input(shape=(max_length,))
    s1 = Embedding(vocab_size, 256, mask_zero=True)(input2)
    s2 = Dropout(0.5)(s1)
    s3 = LSTM(256)(s2)
    # Decoder
    # Merge features from image and text
    d1 = Add()([f2, s3])
    d2 = Dense(256, activation='relu')(d1)
    out = Dense(vocab_size, activation='softmax')(d2)
    # Model Compilation
    model = Model(inputs=[input1, input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Summary
    print(model.summary())
    plot_model(model, to_file=file, show_shapes=True)
    
    return model

#class TransformerBlock(tf.keras.layers.Layer):
    #def __init__():