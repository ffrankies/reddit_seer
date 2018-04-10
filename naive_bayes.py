"""Performs naive Bayes on the data from a given subreddit.
"""
import math

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
import numpy as np

import bag_of_words as bow

MODEL = BernoulliNB()

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

def get_quartiles(scores: []) -> []:
  """Gets the quartile boundaries for the given array of scores.

  Params:
  - scores ([int]): array of integer scores of the posts

  Returns:
  - quartiles ([int]): an array of the 3 quartile boundaries (for q1|q2|q3|q4)
  """
  s = np.array(scores).astype(np.float)
  q1 = math.floor(np.percentile(s, 25))
  q2 = math.floor(np.percentile(s, 50))
  q3 = math.floor(np.percentile(s, 75))

  print("Q1 | Q2  | Q3  | Q4")
  print("   " + str(q1) + "    " + str(q2) + "    " + str(q3))

  return [q1,q2,q3]
# End of get_quartiles()

def extract_targets(scores: [], quartiles: []) -> []:
  """Gets the quartile targets for the given array of scores.

  Params:
  - scores ([int]): array of integer scores of the posts
  - quartiles ([int]): an array of the 3 quartile boundaries (for q1|q2|q3|q4)

  Returns:
  - targets ([int]): array of which quartile each post belongs in
  """
  targets = [None] * len(scores)

  q1 = quartiles[0]
  q2 = quartiles[1]
  q3 = quartiles[2]

  for i in range(0, len(scores)):
    score = int(scores[i])
    if score < q1:
      targets[i] = 1
    elif score < q2:
      targets[i] = 2
    elif score < q3:
      targets[i] = 3
    else:
      targets[i] = 4
  # end for loop

  return targets
# End of extract_targets()

def classify(data_frame: pd.DataFrame):
    """Performs naive Bayes classification on the data in a data frame.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data

    Categories:
    - 1: 1st Quartile
    - 2: 2nd Quartile
    - 3: 3rd Quartile
    - 4: 4th Quartile
    """
    features = extract_features(data_frame)
    num_rows = len(features.index)
    separator = math.floor(0.8 * num_rows)
    train_X = features.loc[:separator].values
    test_X = features.loc[separator:].values

    quartiles = get_quartiles(data_frame.loc[:separator, 'score'].values)

    train_Y = extract_targets(data_frame.loc[:separator, 'score'].values, quartiles)
    test_Y = extract_targets(data_frame.loc[separator:, 'score'].values, quartiles)

    model = MODEL.fit(train_X, train_Y)
    score = model.score(test_X, test_Y)

    print("Done")
    print("R^2 value = (%% of variance explained by model) = {}".format(score))
# End of regress()


if __name__ == "__main__":
    args = bow.parse_arguments()
    data_frame = bow.csv_to_data_frame(args.subreddit)
    classify(data_frame)