import os
import sys
import time

import utils
import model
import data_generator as data_gen

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

def loss_function(loss_obj, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    #pred = tf.squeeze(pred[:, -1:, :]) # (batch_size, max_length, vocab_size) ->(batch_size, vocab_size)
    loss_ = loss_obj(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def train(config):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints = os.path.join(cur_dir, 'checkpoints')
    if len(config.split('.')) < 2:
        cfg_file = os.path.join(cur_dir, 'config/' + config + '.cfg')
    else:
        cfg_file = os.path.join(cur_dir, 'config/' + config)

    cfg =  utils.load_cfg(cfg_file)['default']
    _, _, _, _, _, vocab_size, max_length = utils.prepare()
    file_model = os.path.join(cur_dir, 'checkpoints/' + cfg['model'] + '.png')
    f_shape = (4096,)

    if cfg['model'] == 'simple':
        caption_model = model.simple_caption_model(f_shape, vocab_size, max_length, file_model)
    elif cfg['model'] == 'transformer':
        caption_model = model.transformer_caption_model(f_shape, vocab_size, max_length-1, file_model)
    
    run = cfg['dataset'] + '_' + cfg['backbone'] + '_' + cfg['model']
    run_path = os.path.join(checkpoints, run)
    if not os.path.exists(run_path):
        os.mkdir(run_path)

    checkpoint_callback = ModelCheckpoint(filepath=run_path,
                                            save_weights_only=False,
                                            monitor='loss',
                                            mode='min',
                                            save_best_only=True)
    
    train_gen = data_gen.DataGenerator(vocab_size, max_length)
    #train_dataset = tf.data.Dataset.from_generator(train_gen, (np.ndarray, np.ndarray, np.ndarray))
    #if cfg['model'] == 'transformer':
    #    caption_model.compile(optimizer='adam',
    #                        loss=SparseCategoricalCrossentropy())
    #else:    
    #    caption_model.compile(optimizer='adam',
    #                        loss=CategoricalCrossentropy())

    #lr_schedule = CustomSchedule(f_shape)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=1e-4,
                            decay_steps=10000,
                            decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)
    loss_object = CategoricalCrossentropy()

    #caption_model.fit(x=train_gen,
    #                    use_multiprocessing=False,
                        #workers=6,
    #                    epochs=int(cfg['epochs']),
    #                    callbacks=[checkpoint_callback])
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

                seq_tensor = seq_tensor[:, :-1]
                target = target[:, 1:]
                
                pred = caption_model([f_tensor, seq_tensor])
                loss = loss_function(loss_object, target, pred)
                
                pb_i.add(batch_size, values=[('cross_entropy', loss)])

            gradients = tape.gradient(loss, caption_model.trainable_variables)
            optimizer.apply_gradients(
                (grad, var)
                for (grad, var) in zip(gradients, caption_model.trainable_variables)
                if grad is not None
                )

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
        print('Time taken for 1 epoch {} secs\n'.format(time.time() - start))
        train_gen.on_epoch_end()

if __name__ == "__main__":
    print('<Train>')
    train(sys.argv[1])
    print('<Done>')