

from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize

import pandas as pd
import re
import string

regex = re.compile('[%s]' % re.escape(string.punctuation))
regexb=re.compile('b[\'\"]')
stop_words = set(stopwords.words('english'))


def loadData(path='./data/Combined_News_DJIA.csv'):
    """
      split as per dates: 80 percent train and 20 percent test
      replace nan with empty strings
      make date as an index (helps with time series data)
    """
    data = pd.read_csv(path, parse_dates = True, index_col = 0, verbose =True, keep_default_na=False) 
    test = data['2015-01-02':'2016-07-01']
    train = data['2008-08-08':'2014-12-31']
    return (train, test)

def loadDataCombinedColumns(path='./data/Combined_News_DJIA.csv'):
    """
      Combine all the news headlines(25 columns) into one. 
      All the headlines for any given day is treated as one document
      Split as per dates: 80 percent train and 20 percent test
      replace nan with empty strings
      make date as an index (helps with time series data)
    """
    data = pd.read_csv(path, parse_dates = True, index_col = 0, verbose =True, keep_default_na=False) 
    data_y = data["Label"]
    data_X = data.iloc[:,1:26].apply(lambda headline:cleanString(' '.join(headline)), axis=1)
    
    test_X  = data_X['2015-01-02':'2016-07-01']
    train_X = data_X['2008-08-08':'2014-12-31']
    test_y  = data_y['2015-01-02':'2016-07-01']
    train_y = data_y['2008-08-08':'2014-12-31']
    return (train_X, train_y,  test_X, test_y)


 



def cleanString(sentence):
    """
        get Grams for a  sentence
        Custom Function

    """

    return ' '.join(getGrams(sentence))







def getGrams(sentence):
    """
        get Grams for a  sentence
        Custom Function

    """
    sentence = sentence.lower()
    sentence = regexb.sub('', sentence)
    sentence = regex.sub('', sentence)
    tokens = filter(lambda token: token != '', word_tokenize(sentence))
    #tokens = filter(lambda word: word not in stop_words, tokens)

    
    #return filter(lambda word: word not in stop_words, filter(lambda token: token != '', map(lambda token:regex.sub('', token),map(str.lower, word_tokenize(sentence)))))

    return tokens



def getGramsList(data):

    """
        get unigrams, bigrams, trigrams for list of sentences.
        custom function
    """
    
    unigrams = []
    bigrams = []
    trigrams = []
         
    for sentence in data:
        try:   
            # lower strings
            tokens = map(str.lower, word_tokenize(sentence))
            
            #remove punctuation
            tokens =  map(lambda token:regex.sub('', token), tokens)  
            
            #filter empty strings
            tokens = filter(lambda token: token != '', tokens )
            
    
            # generate bigrams/trigrams            
            bigrams.extend(map('_'.join ,ngrams(tokens, n=2)))
            trigrams.extend(map('_'.join ,ngrams(tokens, n=3)))

            
            # remove stopwords
            tokens = filter(lambda word: word not in stop_words, tokens)
    
            # Unigrams are generated after removal of stopwords
            unigrams.extend(tokens)
        except :
            continue

    return (unigrams, bigrams, trigrams)


def getUnigramFreq(unigrams):
    
    """
        get frequency counts of unigrams. Unigrams is used to generate global vectors.
        Hence must be supplied.
    """    
    
    # init dict with tokens as the keys
    wordFreqDict = dict()
    for word in unigrams:
        if word in wordFreqDict.keys():
            wordFreqDict[word] = wordFreqDict[word] + 1 
        else:
            wordFreqDict[word] = 1

    return wordFreqDict

def getBigramFreq(bigrams):
    

    """
        get frequency counts of bigrams. 
        Hence must be supplied.
    """

    # init dict with tokens as the keys
    wordFreqDict = dict()
    for bigram in bigrams:
        #word = '_'.join(word_tuples)
        if bigram in wordFreqDict.keys():
            wordFreqDict[bigram] = wordFreqDict[bigram] + 1
        else:
            wordFreqDict[bigram] = 1
         
    return wordFreqDict

def getTrigramFreq(trigrams):
    """
        get frequency counts of bigrams. 
        Hence must be supplied.
    """

    # init dict with tokens as the keys
    wordFreqDict = dict()
    for trigram in trigrams:
        #word = '_'.join(word_tuples)
        if trigram in wordFreqDict.keys():
            wordFreqDict[trigram] = wordFreqDict[trigram] + 1
        else:
            wordFreqDict[trigram] = 1
         
    return wordFreqDict



