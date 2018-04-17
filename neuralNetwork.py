from __future__ import division, print_function, absolute_import

from typing import Dict, List, Tuple

from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pickle
import random
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

import bag_of_words as bow
# from tflearn.datasets import imdb
#
# # IMDB Dataset loading
# train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
#                                 valid_portion=0.1)
# trainX, trainY = train
# testX, testY = test
#
# # Data preprocessing
# # Sequence padding
# trainX = pad_sequences(trainX, maxlen=100, value=0.)
# testX = pad_sequences(testX, maxlen=100, value=0.)
# # Converting labels to binary vectors
# trainY = to_categorical(trainY, nb_classes=2)
# testY = to_categorical(testY, nb_classes=2)
#
# # Network building
# net = tflearn.input_data([None, 100])
# net = tflearn.embedding(net, input_dim=10000, output_dim=128)
# net = tflearn.lstm(net, 128, dropout=0.8)
# net = tflearn.fully_connected(net, 2, activation='softmax')
# net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
#                          loss='categorical_crossentropy')
#
# # Training
# model = tflearn.DNN(net, tensorboard_verbose=0)
# model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
#           batch_size=64)


class NeuralNetwork:

    #FIXME do not build ANN here
    def __init__(self):
        pass


    def load_data(self, subreddit: str = 'askscience', column: str = 'title',
        # n_words: int = 2000,
        valid_portion: float = 0.2) -> ((str, int), (str, int)):
        """
        Loads in the specified data set from './data/' with the top n_words
        from the bag_of_words created from the data set.

        Params:
        - subreddit (str): Subreddit of desired data set
        - column (str): Column to use for evaluating score with
        - n_words (int): Number of words to load in from the bag of words
        - valid_portion (float): Portion of data for the validation set

        Returns:
        - Tuple of training and validation sets each consisting of a tuple
            of the input and score
        """
        # Loading dataframe and creating bag of words
        data_frame = bow.csv_to_data_frame(subreddit)
        bag = bow.bag_of_words(data_frame, column)

        # Dictionary of {word: unique index}
        self.word2index = self.get_word_2_index(bag)

        # Getting input and scores
        x = data_frame[column]
        y = data_frame['score']

        # Replacing string data with word ids
        for i,text in enumerate(x):
            x[i] = self.replace_word_2_index(text)

        # Changing scores to int
        for i,score in enumerate(y):
            y[i] = int(score)
        y = self.normalize(y)

        # Create pairs of input and score
        pairs = [(x[i], y[i]) for i in range(len(x))]

        # Break posts into 10 buckets
        sorted(pairs, key=lambda pair: pair[1])

        new_pairs = []
        for pair in pairs[:int(len(pairs)*0.2)]:
            new_pairs.append((pair[0], 0))
        for pair in pairs[int(len(pairs)*0.2):int(len(pairs)*0.4)]:
            new_pairs.append((pair[0], 1))
        for pair in pairs[int(len(pairs)*0.4):int(len(pairs)*0.6)]:
            new_pairs.append((pair[0], 2))
        for pair in pairs[int(len(pairs)*0.6):int(len(pairs)*0.8)]:
            new_pairs.append((pair[0], 3))
        for pair in pairs[int(len(pairs)*0.8):]:
            new_pairs.append((pair[0], 4))
        pairs = new_pairs

        # Shuffling data
        for _ in range(5):
            random.shuffle(pairs)

        # Getting training and testing data
        slice_idx = int(len(pairs) * (1 - valid_portion))
        trainX = [pair[0] for pair in pairs[:slice_idx]]
        trainY = [pair[1] for pair in pairs[:slice_idx]]
        testX = [pair[0] for pair in pairs[slice_idx:]]
        testY = [pair[1] for pair in pairs[slice_idx:]]

        return (trainX, trainY), (testX, testY)


    #TODO document
    def get_word_2_index(self, bag):
        """
        Creates a dictionary of words with each word getting a unique identifier
        """
        # Give each word a unique id
        word2index = {}
        word2index["<unknow>"] = 0
        for i,word in enumerate(bag):
            word2index[word] = i+1 # +1 so that 0 is reserved for TFLearn

        # # Add a base case for words that don't match
        # word2index['<unkw>'] = len(words2index)

        return word2index


    def replace_word_2_index(self, text):
        words = text.split();

        word_set = []
        for word in words:
            if word in self.word2index:
                word_set.append(self.word2index[word])
            else:
                word_set.append(self.word2index["<unknow>"])

        return word_set


    def normalize(self, vals):
        vmax = max(vals)
        vmin = min(vals)
        norm = [(v - vmin)/(vmax-vmin) for v in vals]
        return norm


    #FIXME do returns and document
    def preprocess(self, train, test):
        """
        Flatten pandas DataFrame to a Dictionary of key=word and
        value=unique identifier

        Params:
        - bag (pandas.DataFrame): DataFrame containing a bag of words

        Returns:
        -
        """
        trainX, trainY = train
        testX, testY = test

        # Data preprocessing
        # Sequence padding
        trainX = pad_sequences(trainX, maxlen=40, value=0.)
        testX = pad_sequences(testX, maxlen=40, value=0.)
        # Converting labels to binary vectors
        # trainY = to_categorical(trainY, nb_classes=2)
        # testY = to_categorical(testY, nb_classes=2)
        trainY = to_categorical(trainY, nb_classes=5)
        testY = to_categorical(testY, nb_classes=5)

        return trainX, trainY, testX, testY


    #TODO document
    def create_model(self):
        # Building the model
        net = tflearn.input_data([None, 40])
        net = tflearn.embedding(net, input_dim=len(self.word2index), output_dim=128)
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, 128, activation='tanh')
        net = tflearn.dropout(net, 0.8)
        net = tflearn.fully_connected(net, 128, activation='tanh')
        net = tflearn.dropout(net, 0.8)
        net = tflearn.fully_connected(net, 5, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001,
                                 loss='categorical_crossentropy')

        self.model = tflearn.DNN(net, tensorboard_verbose=0)


    def fit(self, trainX: List[List[int]], trainY: List[float], n_epoch: 10,
        validation_set: Tuple[List[List[int]], List[float]], show_metric: bool,
        batch_size: int):
        """
        Fits the model with the specified data and validation set.

        Params:
        - trainX (List[List[int]]): List of list of ints, with each int
            corresponding to a word in the bag_of_words
        - trainY (List[float]): List of floats corresponding to the score of the
            input data
        - validation_set (Tuple[List[List[int]], List[float]])):
            Validation data set to test the accuracy of the model with
        - show_metric (bool): Whether metric data should be shown
        - batch_size (int): Size of batch of samples to send through the
            neural network at one time
        """
        self.model.fit(trainX, trainY, n_epoch, validation_set, show_metric, batch_size)


    def predict(self, testX):
        # Replacing string data with word ids
        x = [0] * len(testX)
        for i,text in enumerate(testX):
            x[i] = self.replace_word_2_index(text)
        x = pad_sequences(x, maxlen=40, value=0.)

        print(self.model.predict(x))


    def save(self):
        with open('word2index.pickle', 'wb') as f:
            pickle.dump(self.word2index, f)
        self.model.save("model.tflearn")


    def load(self):
        self.word2index = pickle.load(open('word2index.pickle', 'rb'))
        self.create_model()
        self.model.load("model.tflearn")


if __name__ == '__main__':
    nn = NeuralNetwork()
    create_new_model = True
    if create_new_model:
        print("Creating new model")
        train, test = nn.load_data()
        nn.create_model()
        trainX, trainY, testX, testY = nn.preprocess(train, test)
        nn.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=256)
        nn.save()
    else:
        print("Loading previous model")
        nn.load()

    nn.predict([
        "Are magnetic materials stronger than they would otherwise be without their magnetic field?", # 42
        "Why is liquid helium used to cool down superconducting magnets?", # 9
        "Is it possible to insulate a house so well that HVAC isnâ€™t necessary?", # 7
        "How do sugar substitutes like sucralose affect blood insulin levels?", # 583
        "Why doesn't a dark chocolate bar break predictably, despite chocolate's homogeneity and deep grooves in the bar?" # 10231
    ])
