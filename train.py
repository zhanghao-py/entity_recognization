import numpy as np
import os, sys, string

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.cross_validation import train_test_split

import keras as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten

import cPickle as pickle

import utils

def load_train_texts(file_path = 'data/train_data.txt'):
    text_list = []
    with open(file_path, 'r') as f:
        for line in f:
            tokens = string.split(line, '\t')
            sentence, core_term, brand = tokens
            sentence = utils.sentence_preprocessing(sentence)
            core_term = utils.core_term_preprocessing(core_term)
            brand = utils.brand_preprocessing(brand)
            text_list.append([sentence, core_term, brand])
    return text_list

def dnn_training_and_test(X_train, X_test, y_train, y_test):
    y_train_mat = K.utils.np_utils.to_categorical(y_train)
    y_test_mat = K.utils.np_utils.to_categorical(y_test)
    
    print 'model: dnn.'
    model = Sequential()
    model.add(Dense(512, input_shape=(7173,), activation='relu'))
    # model.add(Dense(512, input_shape=(2391,), activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    sgd = K.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss=K.losses.categorical_crossentropy, optimizer=sgd)
    model.fit(X_train, y_train_mat, epochs=3)
    y_train_res = model.predict(X_train)
    y_test_res = model.predict(X_test)

    y_train_pred = np.argmax(y_train_res, axis=1)
    y_test_pred = np.argmax(y_test_res, axis=1)
    y_train_pred_prob = y_train_res[:, 1]
    y_test_pred_prob = y_test_res[:, 1]
    
    evaluate_model(y_train, y_train_pred, y_train_pred_prob, y_test, y_test_pred, y_test_pred_prob)
    return model

def lr_training_and_test(X_train, X_test, y_train, y_test):
    print 'model: logistic regression.'
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]

    evaluate_model(y_train, y_train_pred, y_train_pred_prob, y_test, y_test_pred, y_test_pred_prob)
    return model

def cnn_training_and_test(X_train, X_test, y_train, y_test):
    y_train_mat = K.utils.np_utils.to_categorical(y_train)
    y_test_mat = K.utils.np_utils.to_categorical(y_test)
        
    print 'model: cnn.'
    model = Sequential()
    # model.add(Conv2D(64, kernel_size=(2, 6), activation='relu', input_shape=(3, 2391, 1)))
    model.add(Conv2D(64, kernel_size=(2, 7), activation='relu', input_shape=(3, 128, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(2, activation='softmax'))

    # opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    opt = K.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss=K.losses.categorical_crossentropy, optimizer=opt)
    model.fit(X_train, y_train_mat, epochs=3)
    y_train_res = model.predict(X_train)
    y_test_res = model.predict(X_test)

    y_train_pred = np.argmax(y_train_res, axis=1)
    y_test_pred = np.argmax(y_test_res, axis=1)
    y_train_pred_prob = y_train_res[:, 1]
    y_test_pred_prob = y_test_res[:, 1]

    evaluate_model(y_train, y_train_pred, y_train_pred_prob, y_test, y_test_pred, y_test_pred_prob)
    return model

def evaluate_model(y_train, y_train_pred, y_train_pred_prob, y_test, y_test_pred, y_test_pred_prob):
    print 'train precision: %s, test precision: %s.' % (precision_score(y_train, y_train_pred, average='micro'), precision_score(y_test, y_test_pred, average='micro'))
    print 'train recall: %s, test recall: %s.' % (recall_score(y_train, y_train_pred, average='micro'), recall_score(y_test, y_test_pred, average='micro'))
    print 'train f1-score: %s, test f1-score: %s.' % (f1_score(y_train, y_train_pred, average='micro'), f1_score(y_test, y_test_pred, average='micro'))
    print 'train auc: %s, test auc: %s' % (roc_auc_score(y_train, y_train_pred_prob), roc_auc_score(y_test, y_test_pred_prob))

def train_and_eval(word_dict):
    # Step 1: Feature extraction.
    train_texts = load_train_texts()
    X, y = [], []
    for line in train_texts:
        sentence, core_term, brand = line
        # For LR/DNN model.
        features, core_term_labels, brand_labels = utils.sentence2data(sentence, core_term, brand, word_dict)
        # For cnn-model.
        # features, core_term_labels, brand_labels = utils.sentence2data(sentence, core_term, brand, word_dict, True)
        X.extend(features)
        y.extend(core_term_labels)
        # y.extend(brand_labels)

    print 'feature extraction finished.'
    X = np.array(X)
    # X = np.reshape(X, X.shape + (1, ));
    y = np.array(y)
    print 'X_shape: %s, y_shape: %s' % (X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 2/3: Model training and evaluation.
    # model = lr_training_and_test(X_train, X_test, y_train, y_test)
    model = dnn_training_and_test(X_train, X_test, y_train, y_test)
    # model = cnn_training_and_test(X_train, X_test, y_train, y_test)
    pickle.dump(model, open(core_term_model_file, "wb"))

if __name__ == '__main__':
    global core_term_model_file, brand_model_file
    word_dict = utils.load_word_vector()
    print 'word_dict_size: %s' % len(word_dict)

    core_term_model_file = 'model/core_term_cnn_w2v.model'
    brand_model_file = 'model/brand_dnn.model'

    # Step 1/2/3: Feature extraction, training model and evaluation.
    if not os.path.isfile(core_term_model_file):
        train_and_eval(word_dict)

