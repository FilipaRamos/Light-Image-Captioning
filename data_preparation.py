import os
import string

def load_tokens(file):
    file = open(file, 'r')
    text = file.read()
    file.close()
    return text

def load_descriptions(data):
    map = dict()
    # process descriptions
    for line in data.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        # First token is img id, the rest is description
        img_id, img_desc = tokens[0], tokens[1:]
        # Remove filename from img id
        img_id = img_id.split('.')[0]
        # convert to string
        img_desc = ' '.join(img_desc)
        # There are several captions to the same image
        if img_id not in map:
            map[img_id] = list()
        map[img_id].append(img_desc)
    return map

def process_descriptions(data):
    ''' Clean the captions by
        converting all words to lowercase
        removing all punctuation
        removing all words that are one character length
        removing all words with numbers
        '''
    tb = str.maketrans('','', string.punctuation)
    for key, descs in data.items():
        for i in range(len(descs)):
            desc = descs[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation
            desc = [w.translate(tb) for w in desc]
            # remove one character length
            desc = [word for word in desc if len(word) > 1]
            # remove numbers
            desc = [word for word in desc if word.isalpha()]
            # store
            descs[i] = ' '.join(desc)

def to_vocab(data):
    all = set()
    for key in data.keys():
        [all.update(d.split()) for d in data[key]]
    return all

def save_descriptions(data, file):
    lines = list()
    for key, descs in data.items():
        for desc in descs:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(file, 'w')
    file.write(data)
    file.close()

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(cur_dir, 'data/Flickr8k.token.txt')

    data = load_tokens(file)
    data = load_descriptions(data)
    print("Loaded descriptions>%s" % len(data))
    process_descriptions(data)
    vocab = to_vocab(data)
    print("Vocabulary size>%s" % len(vocab))
    save_descriptions(data, os.path.join(cur_dir, 'data/descriptions.txt'))
