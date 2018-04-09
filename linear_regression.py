"""Performs linear regression on the data from a given subreddit.
"""
import math

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

import bag_of_words as bow


MODEL = LinearRegression()


def extract_features(data_frame: pd.DataFrame) -> pd.DataFrame:
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


def regress(data_frame: pd.DataFrame):
    """Performs linear regression on the data in a data frame.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data
    """
    features = extract_features(data_frame)
    num_rows = len(features.index)
    separator = math.floor(0.8 * num_rows)
    train_X = features.loc[:separator].values
    test_X = features.loc[separator:].values
    train_Y = data_frame.loc[:separator, 'score'].values
    test_Y = data_frame.loc[separator:, 'score'].values
    model = MODEL.fit(train_X, train_Y)
    score = model.score(test_X, test_Y)
    print("R^2 value = (%% of variance explained by model) = {}".format(score))
# End of regress()


if __name__ == "__main__":
    args = bow.parse_arguments()
    data_frame = bow.csv_to_data_frame(args.subreddit)
    regress(data_frame)