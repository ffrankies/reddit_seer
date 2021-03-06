#!/usr/bin/env python3

"""
Analyzes the sentiment of a text. USes the VADER lexicon for determining word
value multiplier. Any words not in the VADER lexicon are given adjectives in
hopes that the new word exists in the lexiconself.

:author Troy Madsen
:author Frank Wanye
:author Kellin McAvoy
:author Andrew Prins
"""

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
import pandas as pd

def tokenizeNonStopWords(s: str):
    """
    Tokenizes a string and returns dict of all non-stopwords

    :param s String to tokenize into dict
    :returns Dict containing all non-stopwords from the input string
    """

    # Tokenize and create a dictionary of all words
    words = word_tokenize(s)
    words = dict(enumerate(words))

    # Remove all stopwords from the dictionary
    stop_words = set(stopwords.words("english"))
    for i in range(len(words)):
        if words[i] in stop_words:
            words.pop(i)

    return words



def analyzeSentiment(sent_text: str):
    """
    Takes in a body of text and returns compound, negative, neutral, and positive sentiment

    :param sent_text Body of text to analyze the sentiment of
    :returns The sentiment score of a text ranging from -1 to 1
    """

    # Get all non-stopwords if a dict of form {index: word}
    nonStopWords = tokenizeNonStopWords(sent_text)

    # All words tokenized
    tokens = word_tokenize(sent_text)

    # Determine if each word is in VADER corpus
    # if in corpus, leave it alone
    # if not in corpus, attempt lemmatizing to see if in corpus
    # Dictionary of all VADER words
    sia = SentimentIntensityAnalyzer()
    lex_dict = sia.make_lex_dict()
    lemmatizer = WordNetLemmatizer()
    for key, val in nonStopWords.items():
        # Check if lexicon has this word
        hasWord = lex_dict.get(val, None)
        if hasWord == None:
            # Word not in lex_dict
            # Lemmatize assuming adjective and replace word
            tokens[key] = lemmatizer.lemmatize(val, 'a')

    # Detokenize to modified sentiment text
    detokenizer = MosesDetokenizer()
    mod_text = detokenizer.detokenize(tokens, return_str=True)

    # Get sentiment scores
    sentiment = sia.polarity_scores(mod_text)

    # Only interested in the compound score
    return sentiment['compound']
# End of analyzeSentiment()

def analyzeSentiments(data_frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Creates a sentiments data frame out of the text in the given column of the data frame.

    Params:
    - data_frame (pd.DataFrame): The data frame containing text data
    - column (str): The name of the column for which to analyze the sentiment

    Returns:
    - sentiments_data_frame (pd.DataFrame): The data frame containing the sentiment for each item in the given column. Note that sentiments are from 0 to 2, with 1 being neutral, 0 being negative, and 2 being positive.
    """
    data_frame_copy = data_frame.copy()
    column_data = data_frame_copy[column]
    column_data = column_data.apply(lambda a: analyzeSentiment(a))
    column_data = column_data.apply(lambda a: a + 1)

    entries = column_data.values.tolist()
    
    res_data_frame = pd.DataFrame(entries, columns=["sentiment"])
    return res_data_frame
# End of analyzeSentiments()


if __name__ == '__main__':
    sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
        "VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
        "VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
        "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
        "VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
        "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
        "The book was good.",         # positive sentence
        "The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
        "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
        "A really bad, horrible book.",       # negative sentence with booster words
        "At least it isn't a horrible book.", # negated negative sentence with contraction
        ":) and :D",     # emoticons handled
        "",              # an empty string is correctly handled
        "Today sux",     #  negative slang handled
        "Today sux!",    #  negative slang with punctuation emphasis handled
        "Today SUX!",    #  negative slang with capitalization emphasis
        "Today kinda sux! But I'll get by, lol", # mixed sentiment example with slang and constrastive conjunction "but"
        "Troy is the coolest person in the world!" # Normal VADER cannot handle this. analyzeSentiment can
    ]

    longs = ["Sometimes I wish I was a doctor. The human body is so interesting! My girlfriend always tells me all about the different things she is doing in her anatomy lab. Sometimes they are disturbing and sometimes they are really interesting. It's always something new that I've never heard of. At least I get to make computers do cool things!",
        "My roommate tries to make spicey memes. They usually fall flat. I'll just tell him to try harder if he want to achieve fame. Fame might be kinda nice to try. I'd like to start-up a company larger than Apple or Microsoft. Running a business would allow me to create my crazy ideas."
    ]

    sia = SentimentIntensityAnalyzer()

    for sentence in sentences:
        print(sentence)
        ss = sia.polarity_scores(sentence)
        print(ss['compound'], analyzeSentiment(sentence))

    print('\n\n')

    for long in longs:
        print(long)
        ss = sia.polarity_scores(long)
        print(ss['compound'], analyzeSentiment(long))

    # Do not run this, trust me
    # VADER: 1.0
    # Lemmatized: 1.0
    # from nltk.corpus import gutenberg
    # print()
    # bible = gutenberg.raw('bible-kjv.txt')
    # print('VADER: {}'.format(sia.polarity_scores(bible)['compound']))
    # print('Lemmatized: {}'.format(analyzeSentiment(bible)))
