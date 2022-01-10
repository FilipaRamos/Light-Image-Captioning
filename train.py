import os
import eval
import utils
import model

from tensorflow.keras.callbacks import ModelCheckpoint

def train():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_model = os.path.join(cur_dir, 'checkpoints/model.png')
    filepath = os.path.join(cur_dir, 'checkpoints')
    
    descriptions, features, _, _, tokenizer, vocab_size, max_length = utils.prepare()

    epochs = 20
    steps = len(descriptions)
    caption_model = model.caption_model(vocab_size, max_length, file_model)
    
    for i in range(epochs):
        generator = utils.data_generator(descriptions, features, tokenizer, max_length, vocab_size)
        caption_model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        caption_model.save(os.path.join(filepath, 'model_' + str(i) + '.h5'))
    
    '''
    train_gen = data_gen.DataGenerator()
    val_gen = data_gen.DataGenerator()
    caption_model.fit_generator(generator=train_gen, 
                                validation_data=val_gen,
                                use_multiprocesing=True,
                                workers=6,
                                epochs=20)
    '''

if __name__ == "__main__":
    print('<Train>')
    train()
    print('<Done>')