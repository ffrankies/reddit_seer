#!/usr/bin/env python3

'''
This script installs the necessary nltk dependencies

@author Troy Madsen
'''

if __name__ == '__main__':
    import nltk
    nltk.download('perluniprops')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('wordnet')
    # nltk.download('gutenberg')
