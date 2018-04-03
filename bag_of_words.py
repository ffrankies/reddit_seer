"""Contains functions for creating bags of words out of lists of text.
"""

import pathlib
import argparse
import csv

import pandas as pd


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
# End of csv_to_data_frame()


def parse_arguments() -> argparse.Namespace:
    """Parses the given command-line arguments.

    Returns:
    - args (argparse.Namespace): Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subreddit', type=str, help='The subreddit from which to get data',
                        default='askreddit')
    args = parser.parse_args()
    print(args)
    return args
# End of parse_arguments()


if __name__ == '__main__':
    args = parse_arguments()
    csv_to_data_frame(args.subreddit)
    print('Hello World!')