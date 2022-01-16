import os
import sys
import time
from layers import TransformerWrapper

import utils
import model
import layers
import data_generator as data_gen

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy

def prep_train(config):
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
        caption_model = layers.TransformerWrapper(
            int(cfg['NUM_LAYERS']), 
            int(cfg['EMBED_DIM']), 
            int(cfg['NUM_HEADS']), 
            int(cfg['DFF']), 
            int(cfg['ROW']), 
            int(cfg['COL']), 
            vocab_size, 
            max_length
        )

    return run_path, cfg, vocab_size, max_length, caption_model

def loss_function(loss_obj, real, pred):
    # Since the decoder generates one word at a time, only the last word matters
    # (the rest is padded)
    loss_ = loss_obj(real, pred)
    batch_size = loss_.shape[0]
    return tf.reduce_sum(loss_) / batch_size

# Labels are the problem? Should be length=34: [0, 0, 0, 4, 6, 1234, 45...] ?
def loss_function2d(loss_obj, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_obj(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def train(config):
    # Prepare variables
    run_path, cfg, vocab_size, max_length, caption_model = prep_train(config)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=1e-4,
                            decay_steps=10000,
                            decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)
    loss_object = CategoricalCrossentropy(reduction='none')
    loss_monitor = 0

    train_gen = data_gen.DataGenerator(vocab_size, max_length, config)
    metric_name = ['cross_entropy']
    batch_size = train_gen.get_batch_size()
    num_samples = int(train_gen.get_max_count() / batch_size)

    for epoch in range(int(cfg['epochs'])):
        start = time.time()
        pb_i = Progbar(num_samples, stateful_metrics=metric_name)
        
        for c in range(0, num_samples, batch_size):
            f_tensor, seq_tensor, target = train_gen.__getitem__(c)

            with tf.GradientTape() as tape:
                # If the batch only has one sample, expand dims to match previous tensors
                if len(seq_tensor.shape) < 2:
                    seq_tensor = tf.expand_dims(seq_tensor, axis=0)
                if len(target.shape) < 2:
                    target = tf.expand_dims(target, axis=0)

                #seq_tensor = seq_tensor[:, :-1]
                #target = target[:, 1:, :]
                
                pred = caption_model([f_tensor, seq_tensor])
                loss = loss_function(loss_object, target, pred)
                #loss = loss_object(target, pred)
                
                pb_i.add(batch_size, values=[('cross_entropy', loss)])

            gradients = tape.gradient(loss, caption_model.trainable_variables)
            optimizer.apply_gradients(
                            (grad, var)
                            for (grad, var) in zip(gradients, caption_model.trainable_variables)
                            if grad is not None)

        print('Epoch {} :>: Loss {:.4f}'.format(epoch + 1, loss.numpy()))
        print('Time taken for 1 epoch {} secs\n'.format(time.time() - start))

        # Monitor loss in order to save model
        if loss.numpy() < loss_monitor or epoch == 0:
            print('Saved Model Weights>...', run_path)
            caption_model.save_weights(run_path)
            loss_monitor = loss.numpy()

        train_gen.on_epoch_end()

def train2D(config):
    # Prepare variables
    run_path, cfg, vocab_size, max_length, caption_model = prep_train(config)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=1e-4,
                            decay_steps=10000,
                            decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)
    loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_monitor = 0

    train_gen = data_gen.DataGenerator(vocab_size, max_length, config)
    metric_name = ['cross_entropy']
    batch_size = train_gen.get_batch_size()
    num_samples = int(train_gen.get_max_count() / batch_size)

    for epoch in range(int(cfg['epochs'])):
        start = time.time()
        pb_i = Progbar(num_samples, stateful_metrics=metric_name)
        
        for c in range(0, num_samples, batch_size):
            f_tensor, seq_tensor, target = train_gen.__getitem__(c)
            pad_mask, look_mask = model.create_masks(seq_tensor, max_length)
            comb_mask = tf.maximum(pad_mask, look_mask)
            
            with tf.GradientTape() as tape:
                preds, _ = caption_model(f_tensor, seq_tensor, True, comb_mask)
                loss = loss_function2d(loss_object, target, preds)
                
                pb_i.add(batch_size, values=[('cross_entropy', loss)])

            gradients = tape.gradient(loss, caption_model.trainable_variables)
            optimizer.apply_gradients(
                            (grad, var)
                            for (grad, var) in zip(gradients, caption_model.trainable_variables)
                            if grad is not None)

        print('Epoch {} :>: Loss {:.4f}'.format(epoch + 1, loss.numpy()))
        print('Time taken for 1 epoch {} secs\n'.format(time.time() - start))

        # Monitor loss in order to save model
        if loss.numpy() < loss_monitor or epoch == 0:
            print('Saved Model Weights>...', run_path)
            caption_model.save_weights(run_path)
            loss_monitor = loss.numpy()

        train_gen.on_epoch_end()

if __name__ == "__main__":
    print('<Train>')
    m = sys.argv[1].split('_')[2]
    if any(char.isdigit() for char in m):
        # Transformer 2D
        train2D(sys.argv[1])
    else:
        train(sys.argv[1])
    print('<Done>')