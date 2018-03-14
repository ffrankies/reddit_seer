"""Retrieves text posts from reddit using the reddit API, and saves them in a csv format.
"""
import pytz  # python timezone
import datetime

import praw


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
    submissions = list()
    returned_submissions = api_request(reddit, subreddit, from_date, to_date)
    while returned_submissions:
        submissions.extend(returned_submissions)
        to_date_timestamp = returned_submissions[-1].created
        to_date = datetime.datetime.fromtimestamp(to_date_timestamp-1)
        returned_submissions = api_request(reddit, subreddit, from_date, to_date)
    print("Got {:d} submissions!".format(len(submissions)))
    return submissions
# End of get_submissions()


if __name__ == '__main__':
    subs = get_submissions("learnpython", datetime.datetime(2018, 2, 1), datetime.datetime(2018, 3, 1))
    print('Hello World!')
