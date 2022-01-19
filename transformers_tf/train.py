import os
import sys
import time
from layers_tf import TransformerWrapper

import utils
import transformers_tf.model as model
import layers_tf
import transformers_tf.data_generator as data_gen

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy

config = 'flickr_inception_transformer2d'
cur_dir = os.path.dirname(os.path.abspath(__file__))
checkpoints = os.path.join(cur_dir, 'checkpoints')
if len(config.split('.')) < 2:
    cfg_file = os.path.join(cur_dir, 'config/' + config + '.cfg')
else:
    cfg_file = os.path.join(cur_dir, 'config/' + config)

cfg =  utils.load_cfg(cfg_file)['default']
_, _, _, _, _, vocab_size, max_length = utils.prepare(cfg)
file_model = os.path.join(cur_dir, 'checkpoints/' + cfg['model'] + '.png')
if cfg['model'] == 'simple' or cfg['model'] == 'transformer':
    f_shape = (int(cfg['f_shape']),)

run = cfg['dataset'] + '_' + cfg['backbone'] + '_' + cfg['model']
run_path = os.path.join(checkpoints, run)
if not os.path.exists(run_path):
    os.mkdir(run_path)
run_path = os.path.join(run_path, run + '.h5')

if cfg['model'] == 'simple':
    caption_model = model.simple_caption_model(
        cfg, 
        f_shape, 
        vocab_size, 
        max_length, 
        file_model
    )
elif cfg['model'] == 'transformer':
    caption_model = model.transformer_caption_model(
        cfg, 
        f_shape, 
        vocab_size, 
        max_length, 
        file_model
    )
elif cfg['model'] == 'transformer2d':
    caption_model = layers_tf.TransformerWrapper(
        int(cfg['NUM_LAYERS']), 
        int(cfg['EMBED_DIM']), 
        int(cfg['NUM_HEADS']), 
        int(cfg['DFF']), 
        int(cfg['ROW']), 
        int(cfg['COL']), 
        vocab_size, 
        max_length
    )

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
   def __init__(self, d_model, warmup_steps=4000):
      super(CustomSchedule, self).__init__()
      self.d_model = d_model
      self.d_model = tf.cast(self.d_model, tf.float32)
      self.warmup_steps = warmup_steps

   def __call__(self, step):
      arg1 = tf.math.rsqrt(step)
      arg2 = step * (self.warmup_steps ** -1.5)
      return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def create_padding_mask(seq):
   seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
   return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
   return mask  # (seq_len, seq_len)

def create_masks_decoder(tar):
   look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
   dec_target_padding_mask = create_padding_mask(tar)
   combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
   return combined_mask


lr_schedule = CustomSchedule(int(cfg['EMBED_DIM']))
optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                epsilon=1e-9)
loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_monitor = 0

# Labels are the problem? Should be length=34: [0, 0, 0, 4, 6, 1234, 45...] ?
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')    

@tf.function
def train_step(img_tensor, seq_tensor):
    desc_in = seq_tensor[:, :-1]
    desc_target = seq_tensor[:, 1:]
    #dec_mask = model.create_masks(seq_tensor)
    dec_mask = create_masks_decoder(desc_in)
    with tf.GradientTape() as tape:
        predictions, _ = caption_model(img_tensor, desc_in,True, dec_mask)
        loss = loss_function(desc_target, predictions)

    gradients = tape.gradient(loss, caption_model.trainable_variables)   
    optimizer.apply_gradients(zip(gradients, caption_model.trainable_variables))
    train_loss(loss)
    train_accuracy(desc_target, predictions)

train_gen = data_gen.DataGenerator(vocab_size, max_length, config)
metric_name = ['cross_entropy']
batch_size = train_gen.get_batch_size()
num_samples = int(train_gen.get_max_count() / batch_size)

for epoch in range(int(cfg['epochs'])):
    start = time.time()
    pb_i = Progbar(num_samples, stateful_metrics=metric_name)
    
    for c in range(0, num_samples, batch_size):
        f_tensor, seq_tensor = train_gen.__getitem__(c)
        train_step(f_tensor, seq_tensor)
            
        pb_i.add(batch_size, values=[('cross_entropy', train_loss.result())])

    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                               train_loss.result(),
                                               train_accuracy.result()))
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    
    # Monitor loss in order to save model
    if train_loss.result() < loss_monitor or epoch == 0:
        print('Saved Model Weights>...', run_path)
        caption_model.save_weights(run_path)
        loss_monitor = train_loss.result()

    train_gen.on_epoch_end()