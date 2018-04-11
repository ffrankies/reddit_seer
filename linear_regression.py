"""Performs linear regression on the data from a given subreddit.
"""
import math
import matplotlib
from datetime import datetime
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import bag_of_words as bow
import sentimentAnalyzer


def extract_features_bow(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Obtains the title and text bags of words from the data frame containing subreddit data.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data

    Returns:
    - features (pd.DataFrame): the training features returned by the data frame
    """
    title_bow = bow.bag_of_words(data_frame, 'title')
    selftext_bow = bow.bag_of_words(data_frame, 'selftext')
    features = pd.concat([title_bow, selftext_bow], axis=1)
    features = features.sample(frac=1).reset_index(drop=True)
    print(features.head())
    return features
# End of extract_features()


def extract_features_sentiment(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculates the sentiment of the text from the data frame containing subreddit data.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data

    Returns:
    - features (pd.DataFrame): the training features returned by the data frame
    """
    texts = data_frame['selftext']
    texts = texts.sample(frac=1).reset_index(drop=True)
    texts = texts.apply(lambda a: sentimentAnalyzer.analyzeSentiment(a))
    scalar = StandardScaler()
    texts = scalar.fit_transform(texts.values.reshape(-1, 1))
    texts = np.squeeze(texts)
    texts = pd.DataFrame(texts)
    print('Texts = ', texts.head())
    return texts
# End of extract_features_sentiment()


def extract_features_tod(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculates the time of day (hour of submission) of the text from the data frame containing subreddit data.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data

    Returns:
    - features (pd.DataFrame): the training features returned by the data frame
    """
    hours = data_frame['created_utc']
    hours = hours.sample(frac=1).reset_index(drop=True)
    hours = hours.apply(lambda a: a.hour)
    scalar = StandardScaler()
    hours = scalar.fit_transform(hours.values.reshape(-1, 1))
    hours = np.squeeze(hours)
    hours = pd.DataFrame(hours)
    print('Hours = ', hours.head())
    return hours
# End of extract_features_tod()


def plot_results(predicted: np.array, actual: np.array, title: str, directory: str, r_squared: float):
    """Plots the linear regression results.

    Params:
    - predicted (np.array): The predicted score values
    - actual (np.array): The actual score values
    - title (str): The title of the plot
    - directory (str): The directory in which to save the plot
    - r_squared (float): The r^2 value of the linear regression model
    """
    print("Plotting the results for regression with {}".format(title))
    data_frame = pd.DataFrame({'predicted_score': predicted, 'actual_score': actual})
    scatterplot = data_frame.plot('actual_score', 'predicted_score', kind='scatter', 
                                  title="{} | R^2 = {:0.2f}".format(title, r_squared))
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
    print('Training the linear regression model')
    model = LinearRegression()
    # print('Shape of train_x = ', np.shape(train_X))
    model = model.fit(train_X, train_Y)
    r_squared = model.score(test_X, test_Y)
    print("R-squared value = {}".format(r_squared))
    predicted = model.predict(test_X)
    # print('predicted: ', predicted[:5])
    # print('test_Y: ', test_Y[:5])
    predicted = scalar.inverse_transform(predicted)
    test_Y = scalar.inverse_transform(test_Y)
    # print('inverse transformed predicted: ', predicted[:5])
    # print('inverse transformed test_Y: ', test_Y[:5])
    plot_results(predicted, test_Y, title, directory, r_squared)
# End of train()


def regress_bow(data_frame: pd.DataFrame, subreddit: str):
    """Performs linear regression using the bags of words as the only features.

    Params:
    - data_frame (pd.DataFrame): The data frame containing subreddit data
    - subreddit (str): The subreddit for which data was extracted
    """
    features = extract_features_bow(data_frame)
    scores = data_frame['score']
    regress(features, scores, subreddit, 'bag_of_words_only')
# End of regress_bow()


def regress_sentiment(data_frame: pd.DataFrame, subreddit: str):
    """Performs linear regression using the sentiment of the body text as the only feature.

    Params:
    - data_frame (pd.DataFrame): The data frame containing subreddit data
    - subreddit (str): The subreddit for which data was extracted
    """
    print('Extracting sentiment')
    features = extract_features_sentiment(data_frame)
    scores = data_frame['score']
    regress(features, scores, subreddit, 'sentiment_only', True)
# End of regress_sentiment()


def regress_tod(data_frame: pd.DataFrame, subreddit: str):
    """Performs linear regression using the time of day (hour of posting) as the only feature.

    Params:
    - data_frame (pd.DataFrame): The data frame containing subreddit data
    - subreddit (str): The subreddit for which data was extracted
    """
    print('Extracting time of day')
    features = extract_features_tod(data_frame)
    scores = data_frame['score']
    regress(features, scores, subreddit, 'time_of_day', True)
# End of regress_tod()


def regress(features, scores, subreddit, title, reshape_train_X=False):
    """Performs linear regression on the data in a data frame.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data
    """
    num_rows = len(features.index)
    separator = math.floor(0.8 * num_rows)
    scalar = StandardScaler()
    Y = scalar.fit_transform(scores.values.reshape(-1, 1))
    Y = np.squeeze(Y)
    # print('Scaled y: ', Y, ' | with len = ', len(Y))
    train_X = features.values[:separator]
    test_X = features.values[separator:]
    train_Y = Y[:separator]
    test_Y = Y[separator:]
    # print('Len train_X = {}, train_Y = {}, test_X = {}, test_Y = {}'.format(len(train_X), len(train_Y), len(test_X), len(test_Y)))
    # print('train_X = {}\ntrain_Y = {}'.format(train_X[:5], train_Y[:5]))
    if reshape_train_X:
        train_X = train_X.reshape(-1, 1)
        test_X = test_X.reshape(-1, 1)
    train(train_X, train_Y, test_X, test_Y, title, "./plots/{}".format(subreddit), scalar)
# End of regress()


if __name__ == "__main__":
    args = bow.parse_arguments()
    data_frame = bow.csv_to_data_frame(args.subreddit)
    regress_bow(data_frame, args.subreddit)
    regress_sentiment(data_frame, args.subreddit)
    regress_tod(data_frame, args.subreddit)
