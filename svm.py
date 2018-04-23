"""Uses SVMs to predict scores for a given subreddit.
"""
import math
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

import bag_of_words as bow  # noqa: E402
import linear_regression  # noqa: E402


def plot_results(predicted: np.array, actual: np.array, title: str, directory: str, r_squared: float):
    """Plots the linear svmion results.

    Params:
    - predicted (np.array): The predicted score values
    - actual (np.array): The actual score values
    - title (str): The title of the plot
    - directory (str): The directory in which to save the plot
    - r_squared (float): The r^2 value of the linear svm model
    """
    print("Plotting the results for svm with {}".format(title))
    data_frame = pd.DataFrame({'predicted_score': predicted, 'actual_score': actual})
    crosstab = pd.crosstab(data_frame.predicted_score, data_frame.actual_score)
    heatmap = sns.heatmap(crosstab, annot=True)
    heatmap.set_title("{} | Accuracy = {:0.2f}".format(title, r_squared))
    figure = heatmap.get_figure()
    figure.savefig("{}/{}.png".format(directory, title))
    matplotlib.pyplot.clf()
# End of plot_results()


def score_buckets(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Transforms scores into buckets based on which percentile they fall into.

    Params:
    - data_frame (pd.DataFrame): Contains post data from a given subreddit

    Returns:
    - modified_data_frame (pd.DataFrame): Contains post data with scores replaced with the given bucket
    """
    df_modified = data_frame.copy()
    scores = df_modified['score'].values
    low = np.percentile(scores, 20)
    med_low = np.percentile(scores, 40)
    med = np.percentile(scores, 60)
    med_high = np.percentile(scores, 80)
    def bucketize(num: int) -> int:
        """Places a number in the given bucket.
        """
        if num > med_high:
            return 4
        elif num > med:
            return 3
        elif num > med_low:
            return 2
        elif num > low:
            return 1
        else:
            return 0
    # End of bucketize()
    df_modified['score'] = df_modified['score'].apply(bucketize)
    return df_modified
# End of score_buckets()


def train(train_X: np.array, train_Y: np.array, test_X: np.array, test_Y: np.array, title: str, directory: str):
    """Trains a linear svmion model, tests it, and plots predicted scores vs actual scores.

    Params:
    - train_X (np.array): Training inputs
    - train_Y (np.array): Training labels
    - test_X (np.array): Test inputs
    - test_Y (np.array): Test labels
    """
    print('Training the SVM model')
    model = LinearSVC()
    model = model.fit(train_X, train_Y)
    accuracy = model.score(test_X, test_Y)
    print("Average accuracy = {}".format(accuracy))
    print("test-x: {}, test-y: {}".format(test_X[:5], test_Y[:5]))
    # print("R^2 values = ", r2_score(np.squeeze(test_X), test_Y))
    predicted = model.predict(test_X)
    # predicted = scalar.inverse_transform(predicted)
    # test_Y = scalar.inverse_transform(test_Y)
    plot_results(predicted, test_Y, title, directory, accuracy)
# End of train()


def svm(features, scores, subreddit, title, reshape_train_X=False):
    """Performs SVM training on the data in a data frame.

    Params:
    - data_frame (pd.DataFrame): data frame containing subreddit data
    """
    num_rows = len(features.index)
    separator = math.floor(0.8 * num_rows)
    Y = scores.values
    train_X = features.values[:separator]
    test_X = features.values[separator:]
    train_Y = Y[:separator]
    test_Y = Y[separator:]
    if reshape_train_X:
        train_X = train_X.reshape(-1, 1)
        test_X = test_X.reshape(-1, 1)
    train(train_X, train_Y, test_X, test_Y, title, "./plots/{}".format(subreddit))
# End of svm()


def svm_bow(data_frame: pd.DataFrame, subreddit: str):
    """Performs SVM training using the bags of words as the only features.

    Params:
    - data_frame (pd.DataFrame): The data frame containing subreddit data
    - subreddit (str): The subreddit for which data was extracted
    """
    print('Extracting bag of words')
    features = linear_regression.extract_features_bow(data_frame)
    scores = data_frame['score']
    svm(features, scores, subreddit, 'svm_bag_of_words_only')
# End of svm_bow()


def svm_sentiment(data_frame: pd.DataFrame, subreddit: str):
    """Performs SVM training using the sentiment as the only feature.

    Params:
    - data_frame (pd.DataFrame): The data frame containing subreddit data
    - subreddit (str): The subreddit for which data was extracted
    """
    print('Extracting sentiment')
    features = linear_regression.extract_features_sentiment(data_frame)
    scores = data_frame['score']
    svm(features, scores, subreddit, 'svm_sentiment_only')
# End of svm_sentiment()


def svm_tod(data_frame: pd.DataFrame, subreddit: str):
    """Performs SVM training using the time of posting as the only features.

    Params:
    - data_frame (pd.DataFrame): The data frame containing subreddit data
    - subreddit (str): The subreddit for which data was extracted
    """
    print('Extracting sentiment')
    features = linear_regression.extract_features_tod(data_frame)
    scores = data_frame['score']
    svm(features, scores, subreddit, 'svm_time_of_day_only')
# End of svm_sentiment()


def svm_all(data_frame: pd.DataFrame, subreddit: str):
    """Performs linear SVM training using all features available.

    Params:
    - data_frame (pd.DataFrame): The data frame containing subreddit data
    - subreddit (str): The subreddit for which data was extracted
    """
    print('Extracting all features')
    bows = linear_regression.extract_features_bow(data_frame)
    sentiment = linear_regression.extract_features_sentiment(data_frame)
    tod = linear_regression.extract_features_tod(data_frame)
    features = pd.concat([bows, sentiment, tod], axis=1)
    scores = data_frame['score']
    svm(features, scores, subreddit, 'svm_all_features')
# End of svm_bow()


if __name__ == "__main__":
    args = bow.parse_arguments()
    data_frame = bow.csv_to_data_frame(args.subreddit)
    data_frame = score_buckets(data_frame)
    svm_bow(data_frame, args.subreddit)
    svm_sentiment(data_frame, args.subreddit)
    svm_tod(data_frame, args.subreddit)
    svm_all(data_frame, args.subreddit)
