"""Performs linear regression on the data from a given subreddit.
"""
import math
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import bag_of_words as bow
import sentimentAnalyzer


def extract_features_bow(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Obtains training features from the data frame containing subreddit data.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data

    Returns:
    - features (pd.DataFrame): the training features returned by the data frame
    """
    title_bow = bow.bag_of_words(data_frame, 'title')
    selftext_bow = bow.bag_of_words(data_frame, 'selftext')
    features = pd.concat([title_bow, selftext_bow], axis=1)
    print(features.head())
    return features
# End of extract_features()


def extract_features_sentiment(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Stuffy stuff
    """
    texts = data_frame['selftext']
    texts = texts.apply(lambda a: sentimentAnalyzer.analyzeSentiment(a))
    print(texts.head())
    return texts
# End of extract_features_sentiment()


def plot_results(predicted, actual, title, directory):
    """plot_results
    """
    data_frame = pd.DataFrame({'predicted_score': predicted, 'actual_score': actual})
    scatterplot = data_frame.plot('actual_score', 'predicted_score', kind='scatter')
    figure = scatterplot.get_figure()
    figure.savefig("{}/{}.png".format(directory, title))
# End of plot_results()


def train(train_X: np.array, train_Y: np.array, test_X: np.array, test_Y: np.array, title: str, directory: str, 
          scalar: StandardScaler):
    """Trains a linear regression model, tests it, and plots predicted scores vs actual scores.

    Params:
    - train_X (np.array): Training inputs
    - train_Y (np.array): Training labels
    - test_X (np.array): Test inputs
    - test_Y (np.array): Test labels
    """
    model = LinearRegression()
    model = model.fit(train_X, train_Y)
    r_squared = model.score(test_X, test_Y)
    print("R-squared value = {}".format(r_squared))
    predicted = model.predict(test_X)
    print('predicted: ', predicted[:5])
    print('test_Y: ', test_Y[:5])
    predicted = scalar.inverse_transform(predicted)
    test_Y = scalar.inverse_transform(test_Y)
    print('inverse transformed predicted: ', predicted[:5])
    print('inverse transformed test_Y: ', test_Y[:5])
    plot_results(predicted, test_Y, title, directory)
# End of train()


def regress_bow(data_frame: pd.DataFrame, subreddit: str):
    """Regress bow
    """
    features = extract_features_bow(data_frame)
    scores = data_frame['score']
    regress(features, scores, subreddit, 'bag_of_words_only')
# End of regress_bow()


def regress_sentiment(data_frame: pd.DataFrame, subreddit: str):
    """Regress sentiment
    """
    features = extract_features_sentiment(data_frame)
    scores = data_frame['score']
    regress(features, scores, subreddit, 'sentiment_only')
# End of regress_sentiment()


def regress(features, scores, subreddit, title):
    """Performs linear regression on the data in a data frame.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data
    """
    num_rows = len(features.index)
    separator = math.floor(0.8 * num_rows)
    scalar = StandardScaler()
    Y = scalar.fit_transform(scores.values.reshape(-1, 1))
    Y = np.squeeze(Y)
    print('Scaled y: ', Y, ' | with len = ', len(Y))
    train_X = features.values[:separator]
    test_X = features.values[separator:]
    train_Y = Y[:separator]
    test_Y = Y[separator:]
    print('Len train_X = {}, train_Y = {}, test_X = {}, test_Y = {}'.format(len(train_X), len(train_Y), len(test_X), len(test_Y)))
    print('train_X = {}\ntrain_Y = {}'.format(train_X[:5], train_Y[:5]))
    train(train_X, train_Y, test_X, test_Y, title, "./plots/{}".format(subreddit), scalar)
    # model = model.fit(train_X, train_Y)
    # score = model.score(test_X, test_Y)
    # print("R^2 value = (%% of variance explained by model) = {}".format(score))
# End of regress()


if __name__ == "__main__":
    args = bow.parse_arguments()
    data_frame = bow.csv_to_data_frame(args.subreddit)
    # regress_bow(data_frame, args.subreddit)
    regress_sentiment(data_frame, args.subreddit)