"""Retrieves text posts from reddit using the reddit API, and saves them in a csv format.
"""
import praw

if __name__ == '__main__':
    reddit = praw.Reddit('reddit_seer')
    reddit.read_only = True
    print('Hello World!')
