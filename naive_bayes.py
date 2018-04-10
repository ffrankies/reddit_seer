"""Performs naive Bayes on the data from a given subreddit.
"""
import math

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV

import bag_of_words as bow

MODEL = BernoulliNB()

# MIN_NEGATIVE = -5 # max score for negative
# MIN_POSITIVE_1 = 5 # First level positive min score
# MIN_POSITIVE_2 = 20
# MIN_POSITIVE_3 = 40
# MIN_POSITIVE_4 = 60
# MIN_POSITIVE_5 = 200


MIN_NEGATIVE = -5 # max score for negative
MIN_POSITIVE_1 = 1 # First level positive min score
MIN_POSITIVE_2 = 10
MIN_POSITIVE_3 = 100
MIN_POSITIVE_4 = 1000
MIN_POSITIVE_5 = 10000



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

def extract_targets(scores: []) -> []:
  targets = [None] * len(scores)

  for i in range(0, len(scores)):
    score = int(scores[i])
    if score > MIN_POSITIVE_5:
      targets[i] = 6
    elif score > MIN_POSITIVE_4:
      targets[i] = 5
    elif score > MIN_POSITIVE_3:
      targets[i] = 4
    elif score > MIN_POSITIVE_2:
      targets[i] = 3
    elif score > MIN_POSITIVE_1:
      targets[i] = 2
    elif score < MIN_NEGATIVE:
      targets[i] = 1
    else:
      targets[i] = 3
  # end for loop

  print(targets)
  return targets
# End of extract_targets()

def classify(data_frame: pd.DataFrame):
    """Performs naive Bayes classification on the data in a data frame.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data

    Categories:
    - O: Negative
    - 1: Neutral
    - 2: Positive 1
    - 3: Positive 2
    - 4: Positive 3
    - 5: Positive 4
    - 6: Positive 5
    """
    features = extract_features(data_frame)
    num_rows = len(features.index)
    separator = math.floor(0.8 * num_rows)
    train_X = features.loc[:separator].values
    test_X = features.loc[separator:].values

    train_Y = extract_targets(data_frame.loc[:separator, 'score'].values)
    test_Y = extract_targets(data_frame.loc[separator:, 'score'].values)

    model = MODEL.fit(train_X, train_Y)
    score = model.score(test_X, test_Y)

    print("Done")
    print("R^2 value = (%% of variance explained by model) = {}".format(score))
# End of regress()


if __name__ == "__main__":
    args = bow.parse_arguments()
    data_frame = bow.csv_to_data_frame(args.subreddit)
    classify(data_frame)