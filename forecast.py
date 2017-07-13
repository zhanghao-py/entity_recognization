import numpy as np
import os, sys, string
import cPickle as pickle

import utils

def load_test_texts(file_path = 'data/test_data.txt'):
    text_list = []
    with open(file_path, 'r') as f:
        for line in f:
            sentence = utils.sentence_preprocessing(line)
            text_list.append(sentence)
    return text_list

if __name__ == '__main__':
    global core_term_model_file, brand_model_file
    core_term_model_file = 'model/core_term_dnn.model'
    brand_model_file = 'model/brand_dnn.model'

    word_dict = utils.load_word_vector()

    if not os.path.isfile(core_term_model_file):
        print 'Please training core_term model first and more details in README.md.'
        exit(0)

    if not os.path.isfile(brand_model_file):
        print 'Please training brand model first and more details in README.md.'
        exit(0)

    test_texts = load_test_texts()
    core_term_model = pickle.load(open(core_term_model_file, "rb"))
    brand_model = pickle.load(open(brand_model_file, "rb"))
    for sentence in test_texts:
        tokens = string.split(sentence, ' ')
        X = utils.sentence2fea(sentence, word_dict)

        # CoreTerm Predication.
        core_term_res = core_term_model.predict(np.array(X))
        core_term_pred = np.argmax(core_term_res, axis=1)
        core_term_indices = np.where(core_term_pred > 0)[0]
 
        # Brand Predication.
        brand_res = brand_model.predict(np.array(X))
        brand_pred = np.argmax(brand_res, axis=1)
        brand_indices = np.where(brand_pred > 0)[0]
 
        print 'Sentence: [%s], CoreTerm: [%s], Band: [%s]' % (sentence, utils.indices2str(tokens, core_term_indices), utils.indices2str(tokens, brand_indices))

