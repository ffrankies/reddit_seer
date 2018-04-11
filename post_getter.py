"""Retrieves text posts from reddit using the reddit API, and saves them in a csv format.

Authors: Frank Wanye, Kellin McAvoy, Andrew Prins, Troy Madsen
"""
import datetime
import argparse
import requests

import pathlib
import csv


TODAY = datetime.datetime.utcnow()
ONE_MONTH_DELTA = datetime.timedelta(days=30)
MINUS_ONE_MONTH = TODAY - ONE_MONTH_DELTA


def submissionFilter(submission):
    """Filters the given submission to make sure it has text in its body.

    Params:
    - submission (dict): The submission to be filtered

    Returns:
    - contains_text (bool): True if the submission contains text, false otherwise
    """
    if (not submission['is_self'] or 'selftext' not in submission.keys() or not submission['selftext']
            or submission['selftext'] == '[deleted]' or submission['selftext'] == '[removed]'):
        return False
    return True
# End of submissionFilter()


def api_request(subreddit: str, from_date: int, until_date: int) -> list:
    """Uses the reddit API to get the results of a single API request.

    Params:
    - reddit (praw.Reddit): The Reddit instance to use to run the quest
    - subreddit (str): The subreddit on which to run the request
    - from_date (int): UTC timestamp for the date from which to start looking for submissions
    - until_date (int): UTC timestamp for the date at which to stop looking for submissions

    Returns:
    - submissions (list<dict>): The list of submissions found
    """
    print("from: {} to: {}".format(
        datetime.datetime.fromtimestamp(from_date), datetime.datetime.fromtimestamp(until_date)))
    response = requests.get(
        "https://api.pushshift.io/reddit/search/submission/?subreddit={}&after={}&before={}&size=500&sort=asc".format(
            subreddit, from_date, until_date-1
        ))
    response = response.json()
    submissions = response['data']
    submissions = filter(submissionFilter, submissions)
    submissions = list(submissions)
    return submissions
# End of api_request()


def get_submissions(subreddit: str, from_date: datetime.datetime, until_date: datetime.datetime) -> list:
    """Uses the reddit API to find submissions between two dates.

    Params:
    - from_date (int): UTC timestamp for the date from which to start looking for submissions
    - until_date (int): UTC timestamp for the date at which to stop looking for submissions
    """
    returned_submissions = api_request(subreddit, from_date, until_date)
    num_submissions = 0
    original_from_date = datetime.datetime.fromtimestamp(from_date)
    while returned_submissions:
        num_submissions += len(returned_submissions)
        latest_submission_timestamp = returned_submissions[-1]['created_utc']
        print('Got {} submissions from {} until {}'.format(num_submissions, 
              original_from_date,
              datetime.datetime.fromtimestamp(latest_submission_timestamp)))
        submissions_to_csv(subreddit, returned_submissions)
        returned_submissions = api_request(subreddit, latest_submission_timestamp, until_date)
    print("Got {:d} submissions!".format(num_submissions))
# End of get_submissions()


def submissions_to_csv(subreddit: str, submissions: list):
    """Saves the submissions as a csv file in data/<subreddit>/submissions.csv.

    Params:
    - subreddit (str): The subreddit for which the submissions were obtained
    - submissions (list<praw.Submission>): The submissions returned from the reddit API
    """
    directory_path = pathlib.Path("./data/{}".format(subreddit))
    directory_path.mkdir(parents=True, exist_ok=True)
    file_path = directory_path / 'submissions.csv'
    new_file = True
    if file_path.is_file():  # pylint: disable=E1101
        new_file = False
    with file_path.open('a', encoding="utf-8") as csv_file:  # pylint: disable=E1101
        csv_writer = csv.writer(csv_file)
        if new_file:  # Write headings
            csv_writer.writerow(
                ['title', 'score', 'num_comments', 'over_18', 'created_utc', 'selftext'])
        for submission in submissions:
            csv_writer.writerow([submission['title'], submission['score'],
                                submission['num_comments'], submission['over_18'], 
                                datetime.datetime.fromtimestamp(submission['created_utc']), 
                                submission['selftext'].replace('\n', "\\n")])
# End of submissions_to_csv()


def date_type(date_arg: str) -> datetime.datetime:
    """Returns the date associated with a given string of the format YYYY-MM-DD, where the month and day are optional.

    Params:
    - date_arg (str): The date argument passed in by the user

    Returns:
    - date (datetime.datetime): The datetime object made from the argument
    """
    try:
        date = datetime.datetime.strptime(date_arg, "%Y-%m-%d")
    except ValueError:
        try:
            date = datetime.datetime.strptime(date_arg, "%Y-%m")
        except ValueError:
            date = datetime.datetime.strptime(date_arg, "%Y")
    return date
# End of date_type()


def parse_arguments() -> argparse.Namespace:
    """Parses the given command-line arguments.

    Returns:
    - args (argparse.Namespace): Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subreddit', type=str, help='The subreddit from which to get data',
                        default='askreddit')
    parser.add_argument('-f', '--from_date', type=date_type,
                        help='The date from which to start getting data (format = YYYY-MM-DD)',
                        default=datetime.datetime(2018, 2, 1))
    parser.add_argument('-u', '--until_date', type=date_type,
                        help='The date until which to get data (format = YYYY-MM-DD)',
                        default=datetime.datetime(2018, 3, 1))
    args = parser.parse_args()
    print(args)
    return args
# End of parse_arguments()


def to_utc(date: datetime.datetime) -> int:
    """Converts the date to a utc timestamp so it plays nice with the pushshift API.

    Params:
    - date (datetime.datetime): The date in local time

    Returns:
    - utc_timestamp (int): The utc timestamp
    """
    timestamp = date.timestamp()
    utc_date = datetime.datetime.utcfromtimestamp(timestamp)
    utc_timestamp = utc_date.timestamp()
    utc_timestamp = int(utc_timestamp)
    return utc_timestamp
# End of to_utc()


if __name__ == '__main__':
    args = parse_arguments()
    utc_from_date = to_utc(args.from_date)
    utc_until_date = to_utc(args.until_date)
    get_submissions(args.subreddit, utc_from_date, utc_until_date)
