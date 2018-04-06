#!/usr/bin/env python3

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer

"""
Tokenizes a string and returns dict of all non-stopwords

@param s String to tokenize into dict
@return Dict containing all non-stopwords from the input string
"""
def tokenizeNonStopWords(s):
    # Tokenize and create a dictionary of all words
    words = word_tokenize(s)
    words = dict(enumerate(words))

    # Remove all stopwords from the dictionary
    stop_words = set(stopwords.words("english"))
    for i in range(len(words)):
        if words[i] in stop_words:
            words.pop(i)

    return words



#FIXME add the return statement
"""
Takes in a body of text and returns compound, negative, neutral, and positive sentiment

@param sent_text Body of text to analyze the sentiment of
@return
"""
def analyzeSentiment(sent_text):
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

    sid = SentimentIntensityAnalyzer()

    for sentence in sentences:
        print(sentence)
        ss = sid.polarity_scores(sentence)
        print(ss['compound'], analyzeSentiment(sentence))
