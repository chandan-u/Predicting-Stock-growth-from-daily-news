import gensim
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


import multiprocessing
cores = multiprocessing.cpu_count()



#import sys
#sys.path.append('../')
import dataloader as dtl

train_X, train_y, test_X, test_y  = dtl.loadDataCombinedColumns("data/Combined_News_DJIA.csv")


# train word2vec:  Takes tokenized sentences list as input
# sg=0 refers cbow, sg=1 refers skip gram model
model = gensim.models.Word2Vec(sg=0, sentences= map(word_tokenize, train_X), size=300, window=5)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

# bigram factorizer
# in bigram factorizer words such as new york time etc have their context intact. Vectors are for phrases.




def getAvgDocVectors(w2v, train_X, test_X):
    """
        supply word vectors and list of docs
        get the average vector for each doc;
    """
    w2v_words = w2v.keys()
    vector_size = w2v.values()[0].shape[0]

    train_sentences_tokens = map(word_tokenize, train_X)  # unigram token for each sentence
    test_sentences_tokens = map(word_tokenize, test_X)

    train_docvectors = np.empty((train_X.shape[0], vector_size), np.float32)
    test_docvectors = np.empty((test_X.shape[0], vector_size), np.float32)

    for index, sentence_tokens in enumerate(train_sentences_tokens):

        train_docvectors[index] = np.mean([ w2v[str(word)]  for word in sentence_tokens if word in w2v_words])

    for index, sentence_tokens in enumerate(test_sentences_tokens):

        test_docvectors[index] = np.mean( [ w2v[str(word)] for word in sentence_tokens if word in w2v_words])
        
    return (train_docvectors, test_docvectors)



def getTfIdfWeightedAvgDocVectors(w2v, train_X, test_X):
    """
        supply word vectors and list of docs
        get the average vector for each doc; with tf-idf weights
        train_X : to generate actual tf-idf weights for each doc

    """
    # word vectors: skip gram
    w2v_words = w2v.keys()
    vector_size = w2v.values()[0].shape[0]

    
    # tfidf weights
    # initialize tfIdfVectors: lowercased and stop_words removed
    tfIdfVectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))
    train_X_tfidf = tfIdfVectorizer.fit_transform(train_X).toarray()
    test_X_tfidf  = tfIdfVectorizer.transform(test_X).toarray()
    tf_idf_dict = tfIdfVectorizer.vocabulary_
    tf_idf_vocab = tf_idf_dict.keys()

    train_sentences_tokens = map(word_tokenize, train_X)  # unigram token for each sentence
    test_sentences_tokens = map(word_tokenize, test_X)

    # init doc vectors
    train_docvectors = np.empty((train_X.shape[0], vector_size), float)
    test_docvectors = np.empty((test_X.shape[0], vector_size), float)

    for index, sentence_tokens in enumerate(train_sentences_tokens): 
        
        train_docvectors[index] = np.mean([w2v[word]*train_X_tfidf[index, tf_idf_dict[word]]  for word in sentence_tokens if word in w2v_words if word in tf_idf_vocab ])
        
    for index, sentence_tokens in enumerate(test_sentences_tokens):

        test_docvectors[index] = np.mean([w2v[word]*test_X_tfidf[index, tf_idf_dict[word]]  for word in sentence_tokens if word in w2v_words if word in tf_idf_vocab])
        
    return (train_docvectors, test_docvectors)


w2v_train_X, w2v_test_X = getAvgDocVectors(w2v, train_X, train_y)
w2v_tfIdf_train_X, w2v_tf_Idf_test_X = getTfIdfWeightedAvgDocVectors(w2v, train_X, train_y)



np.savetxt('../data/w2v_train_X.txt')
np.savetxt('../data/w2v_test_X.txt')
np.savetxt('../data/w2v_tfidf_train_X.txt')
np.savetxt('../data/w2v_tfidf_test_X.txt')

        
