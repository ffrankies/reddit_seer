#!/usr/bin/env python3

'''
This script installs the necessary nltk.vader dependencies

@author Troy Madsen
'''

if __name__ == '__main__':
    import nltk
    nltk.download('vader_lexicon')
