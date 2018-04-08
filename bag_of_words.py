"""Contains functions for creating bags of words out of lists of text.
"""

import pathlib
import argparse
import csv
import itertools

import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer


class ColumnIndexes:
    title = 0
    score = 1
    ups = 2
    downs = 3
    num_comments = 4
    over_18 = 5
    created_utc = 6
    selftext = 7


def read_csv(subreddit: str) -> list:
    """Reads the csv file containing reddit posts, filters them to make sure they are valid, and stores them to a list.

    Params:
    - subreddit (str): The subreddit whose posts were saved

    Returns:
    - posts (list<list<str>>): The list of valid post metadata
    """
    posts = []
    submissions = pathlib.Path("./data/{}/submissions.csv".format(subreddit))
    if not submissions.is_file():
        raise Exception("No data for the given subreddit exists: {}".format(subreddit))
    with submissions.open('r') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_reader.__next__()  # Skip over column names
        for row in csv_reader:
            if not row:  # If row is empty
                continue
            if len(row) != 8:  # If there is an extra or missing row
                continue
            if not row[ColumnIndexes.selftext]:  # If there is no post text
                continue
            # print(row)
            row[ColumnIndexes.selftext] = row[ColumnIndexes.selftext].replace('\\n', '\n')
            posts.append(row)
    print("Got {} valid posts!".format(len(posts)))
    return posts
# End of read_csv()


def csv_to_data_frame(subreddit: str) -> pd.DataFrame:
    """Reads csv data from the given file, filters out the 'wrong' rows, and stores it in a data frame.

    Params:
    - subreddit (str): The name of the subreddit whose posts are being analyzed

    Returns:
    - data_frame (pd.DataFrame): The data frame containing the post data
    """
    csv_data = read_csv(subreddit)
    data_frame = pd.DataFrame(csv_data)
    data_frame.columns = ['title', 'score', 'ups', 'downs', 'num_comments', 'over_18', 'created_utc', 'selftext']
    print(data_frame.head())
    return data_frame
# End of csv_to_data_frame()


def parse_arguments() -> argparse.Namespace:
    """Parses the given command-line arguments.

    Returns:
    - args (argparse.Namespace): Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subreddit', type=str, help='The subreddit from which to get data',
                        default='lifeofnorman')
    args = parser.parse_args()
    print(args)
    return args
# End of parse_arguments()


def preprocess_text(data_frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Preprocesses the text in the given column of a data frame and returns it as a new data frame with just 
    that column. 

    Params:
    - data_frame (pd.DataFrame): The data frame containing text data
    - column (str): The name of the column for which to create the bag of words

    Returns:
    - modified_data_frame (pd.DataFrame): The data frame with the preprocessed column
    """
    column_data = data_frame[column]
    column_data = column_data.apply(lambda a: a.lower())
    column_data = column_data.apply(nltk.word_tokenize)
    column_data = column_data.apply(lambda a: [word for word in a if len(word) > 1])
    column_data = column_data.apply(lambda a: [word for word in a if not word.isnumeric()])
    column_data = column_data.apply(lambda a: " ".join(a))
    return column_data
# End of preprocess_text()


def bag_of_words(data_frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Creates a bag of words out of the text in the given column of the data frame.

    Params:
    - data_frame (pd.DataFrame): The data frame containing text data
    - column (str): The name of the column for which to create the bag of words

    Returns:
    - bags (pd.DataFrame): The data frame containing the bags of words for each item in the given column
    """
    data_frame_copy = data_frame.copy()
    column_data = preprocess_text(data_frame_copy, column)
    entries = column_data.values.tolist()
    countVectorizer = CountVectorizer()
    bag = countVectorizer.fit_transform(entries)
    bags = bag.toarray()
    bags_data_frame = pd.DataFrame(bags, columns=countVectorizer.vocabulary_.keys())
    print(bags_data_frame.head())
    return bags_data_frame
# End of bag_of_words()


if __name__ == '__main__':
    args = parse_arguments()
    data_frame = csv_to_data_frame(args.subreddit)
    bag_of_words(data_frame, 'title')
    print('Hello World!')