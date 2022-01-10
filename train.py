import os
import utils
import model

from tensorflow.keras.callbacks import ModelCheckpoint

def prepare_train():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_train = os.path.join(cur_dir, 'data/Flickr_8k.trainImages.txt')
    file_test = os.path.join(cur_dir, 'data/Flickr_8k.devImages.txt')
    file_d = os.path.join(cur_dir, 'data/descriptions.txt')
    file_img = os.path.join(cur_dir, 'data/features.pkl')

    data = utils.load_set(file_train)
    print('Train dataset>%d' % len(data))
    descriptions = utils.load_clean_descriptions(file_d, data)
    print("Train descriptions>%d" % len(descriptions))
    features = utils.load_img_features(file_img, data)
    print("Train features size>%d" % len(features))

    tokenizer = utils.set_tokenizer(descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocab size>%d' % vocab_size)
    max_length = utils.max_length(descriptions)
    print('Max length>%d' % max_length)
    
    # Train Sequences
    x1train, x2train, ytrain = utils.create_seqs(tokenizer, max_length, descriptions, features, vocab_size)

    data_test = utils.load_set(file_test)
    print('Test dataset>%d' % len(data_test))
    descriptions_test = utils.load_clean_descriptions(file_d, data_test)
    print("Test descriptions>%d" % len(descriptions_test))
    features_test = utils.load_img_features(file_img, data_test)
    print("Test features size>%d" % len(features_test))

    # Test Sequences
    x1test, x2test, ytest = utils.create_seqs(tokenizer, max_length, descriptions_test, features_test, vocab_size)

    return x1train, x2train, ytrain, x1test, x2test, ytest, vocab_size, max_length

def train():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_model = os.path.join(cur_dir, 'misc/model.png')
    
    x1train, x2train, ytrain, x1test, x2test, ytest, vocab_size, max_length = prepare_train()

    filename = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    filepath = os.path.join(cur_dir, 'checkpoints/' + filename)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model = model.caption_model(vocab_size, max_length, file_model)
    model.fit([x1train, x2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([x1test, x2test], ytest))

if __name__ == "__main__":
    train()
    print('<Done>')