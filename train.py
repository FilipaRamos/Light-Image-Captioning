import os
import sys
import utils
import model

import data_generator as data_gen

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

def train(config):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_model = os.path.join(cur_dir, 'checkpoints/model.png')
    checkpoints = os.path.join(cur_dir, 'checkpoints')
    cfg_file = os.path.join(cur_dir, 'config/' + config)

    cfg =  utils.load_cfg(cfg_file)
    _, _, _, _, _, vocab_size, max_length = utils.prepare()

    caption_model = model.simple_caption_model(vocab_size, max_length, file_model)
    
    '''for i in range(epochs):
        generator = utils.data_generator(descriptions, features, tokenizer, max_length, vocab_size)
        caption_model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        caption_model.save(os.path.join(filepath, 'model_' + str(i) + '.h5'))
    '''
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
    #val_gen = data_gen.DataGenerator(vocab_size, max_length, train=False)
    caption_model.compile(optimizer='adam',
                            loss=CategoricalCrossentropy())
    caption_model.fit(x=train_gen,
                        use_multiprocessing=True,
                        workers=6,
                        epochs=cfg['epochs'],
                        callbacks=[checkpoint_callback])
    

if __name__ == "__main__":
    print('<Train>')
    train(sys.argv[1])
    print('<Done>')