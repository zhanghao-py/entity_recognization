import string

def sentence_preprocessing(sentence):
    sentence = string.rstrip(string.lstrip(sentence))
    sentence = string.replace(sentence, "/", "")
    sentence = string.replace(sentence, ",", " ")
    sentence = string.replace(sentence, "  ", " ")
    sentence = string.replace(sentence, "MicroSD", "Micro SD")
    return sentence

def core_term_preprocessing(core_term):
    core_term = string.rstrip(string.lstrip(core_term))
    core_term = string.replace(core_term, "MicroSD", "Micro SD")
    return core_term

def brand_preprocessing(brand):
    return string.rstrip(string.lstrip(brand))

def indices2str(tokens, indices):
    unique_tokens = set([tokens[i] for i in indices])
    return string.join(unique_tokens, ' ')

def subsentence2fea(tokens, i, padding, word_dict, column_fea = False):
    tokens_size = len(tokens)
    fea_i = []

    # index at i-1, i, i+1
    for j in xrange(i-padding, i+padding+1):
        if j < 0:
            key = '<s>'
        elif j > tokens_size - 1:
            key = '</s>'
        else:
            key = tokens[j]

        fea_dim = len(word_dict)
        # BoW feature
        tokens_fea = [0] * fea_dim
        if key in word_dict:
            tokens_fea[word_dict[key]] = 1
        
        # Word2Vec feature
        # tokens_fea = word_dict[key] if key in word_dict else [0] * 128 

        if not column_fea:
            fea_i.extend(tokens_fea)
        else:
            fea_i.append(tokens_fea)
    return fea_i

def sentence2fea(sentence, word_dict, column_fea = False, win = 3):
    sentence = string.lower(sentence)
    
    tokens = string.split(sentence, ' ')
    tokens_size = len(tokens)
    padding = int(win/2)

    features = []
    for i in xrange(tokens_size):
        # index at i-1, i, i+1
        features.append(subsentence2fea(tokens, i, padding, word_dict, column_fea))
    return features

def sentence2data(sentence, core_term, brand, word_dict, column_fea = False, win = 3):
    sentence = string.lower(sentence)
    core_term = string.lower(core_term)
    brand = string.lower(brand)
    
    tokens = string.split(sentence, ' ')
    tokens_size = len(tokens)
    padding = int(win/2)

    core_term_tokens = string.split(core_term, ' ')
    brand_tokens = string.split(brand, ' ')

    features, core_term_labels, brand_labels = [], [], []
    for i in xrange(tokens_size):
        # index at i-1, i, i+1
        features.append(subsentence2fea(tokens, i, padding, word_dict, column_fea))

        core_term_label = 1 if tokens[i] in core_term_tokens else 0
        brand_label = 1 if tokens[i] in brand_tokens else 0
        core_term_labels.append(core_term_label)
        brand_labels.append(brand_label)
    return features, core_term_labels, brand_labels

def line2fea(line):
    tokens = string.split(line, ' ')
    word = tokens[0]
    fea = []
    for i in xrange(1, len(tokens)):
        try:
            fea.append(float(tokens[i]))
        except ValueError, e:
            continue
    return word, fea

def load_word_vector(file_path = 'model/vocab.txt'):
# def load_word_vector(file_path = 'model/w2v_vec.txt'):
    word_dict = dict()
    first_line = True
    index = 0
    with open(file_path, 'r') as f:
        num_sample, fea_dim = 0, 0
        for line in f:
            line = string.strip(line)
            word = string.lower(line)
            word_dict[word] = index
            index += 1
            '''
            if first_line:
                first_line = False
                tokens = string.split(line, ' ')
                num_sample, fea_dim = int(tokens[0]), int(tokens[1])
                continue
            else:
                word, fea = line2fea(line)
                word = string.lower(word)
                assert len(fea) == fea_dim
                word_dict[word] = fea
            '''
    return word_dict


