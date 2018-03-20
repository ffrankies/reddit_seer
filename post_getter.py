"""Retrieves text posts from reddit using the reddit API, and saves them in a csv format.

Authors: Frank Wanye, Kellin McAvoy, Andrew Prins, Troy Madsen
"""
import datetime
import argparse

import praw
import pathlib
import csv


TODAY = datetime.datetime.utcnow()
ONE_MONTH_DELTA = datetime.timedelta(days=30)
MINUS_ONE_MONTH = TODAY - ONE_MONTH_DELTA


def api_request(reddit: praw.Reddit, subreddit: str, from_date: datetime.datetime, to_date: datetime.datetime) -> list:
    """Uses the reddit API to get the results of a single API request.

    Params:
    - reddit (praw.Reddit): The Reddit instance to use to run the quest
    - subreddit (str): The subreddit on which to run the request
    - from_date (datetime.datetime): Date from which to start looking for submissions
    - to_date (datetime.datetime): Date at which to stop looking for submissions

    Returns:
    - submissions (list<praw.Submission>): The list of submissions found
    """
    from_timestamp = int(from_date.timestamp())
    to_timestamp = int(to_date.timestamp())
    query = "is_self:1 timestamp:{}..{}".format(from_timestamp, to_timestamp)
    results = reddit.subreddit(subreddit).search(query, sort="new", syntax="cloudsearch", limit=1000)
    submissions = list(results)
    return submissions
# End of api_request()


def get_submissions(subreddit: str, from_date: datetime.datetime, to_date: datetime.datetime) -> list:
    """Uses the reddit API to find submissions between two dates.

    Params:
    - from_date (datetime.datetime): Date from which to start looking for submissions
    - to_date (datetime.datetime): Date at which to stop looking for submissions

    Returns:
    - submissions (list<praw.Submission>): The list of submissions found
    """
    reddit = praw.Reddit('reddit_seer')
    returned_submissions = api_request(reddit, subreddit, from_date, to_date)
    num_submissions = 0
    while returned_submissions:
        num_submissions += len(returned_submissions)
        print('Got {} submissions from {} until {}'.format(num_submissions, 
              datetime.datetime.fromtimestamp(returned_submissions[0].created),
              datetime.datetime.fromtimestamp(returned_submissions[-1].created)))
        submissions_to_csv(subreddit, returned_submissions)
        to_date_timestamp = returned_submissions[-1].created
        to_date = datetime.datetime.fromtimestamp(to_date_timestamp-1)
        returned_submissions = api_request(reddit, subreddit, from_date, to_date)
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
    with file_path.open('a') as csv_file:  # pylint: disable=E1101
        csv_writer = csv.writer(csv_file)
        if new_file:  # Write headings
            csv_writer.writerow(
                ['title', 'score', 'ups', 'downs', 'num_comments', 'over_18', 'created_utc', 'selftext'])
        for submission in submissions:
            csv_writer.writerow([submission.title, submission.score, submission.ups, submission.downs, 
                                submission.num_comments, submission.over_18, 
                                datetime.datetime.fromtimestamp(submission.created_utc), submission.selftext])
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
                        help='The date from which to start getting data (format = YYYY_MM_DD)',
                        default=datetime.datetime(2018, 2, 1))
    parser.add_argument('-u', '--until_date', type=date_type,
                        help='The date until which to get data (format = YYYY_MM_DD)',
                        default=datetime.datetime(2018, 3, 1))
    args = parser.parse_args()
    print(args)
    return args
# End of parse_arguments()


if __name__ == '__main__':
    args = parse_arguments()
    get_submissions(args.subreddit, args.from_date, args.until_date)
