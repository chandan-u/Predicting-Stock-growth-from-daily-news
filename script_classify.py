#from __future__ import division  # floating point division

import math
import numpy as np


import dataloader as dtl
import algorithms as algs
import utilities as utils
 
from sklearn.feature_extraction.text import CountVectorizer

from wordEmbeddings.w2v import getAvgDocVectors, getTfIdfWeightedAvgDocVectors




if __name__ == '__main__':
   


    train_X, train_y, test_X, test_y  = dtl.loadDataCombinedColumns()


    
    # bag of words vectorizers:
    UnigramVectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1), stop_words='english')
    BigramVectorizer = CountVectorizer(lowercase=True, ngram_range=(2, 2), stop_words='english')

    # compute word2vec features ( vecotrs capture semantic nature of words)
    # model = gensim.models.Word2Vec(sg=0, sentences= map(word_tokenize, train_X), size=300, window=5)
    # w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    # w2v_train_X, w2v_test_X = getAvgDocVectors(w2v, train_X, train_y)
    # w2v_tfIdf_train_X, w2v_tf_Idf_test_X = getTfIdfWeightedAvgDocVectors(w2v, train_X, train_y)

    # load computed features (saves time: rather than execute everytime)
    
    w2v_train_X, w2v_test_X = np.loadtxt("data/w2v_train_X.txt"), np.loadtxt("data/w2v_test_X.txt")
    w2v_tfIdf_train_X, w2v_tf_Idf_test_X = np.loadtxt("data/w2v_tfIdf_train_X.txt"), np.loadtxt("data/w2v_tfIdf_test_X.txt")
    
    featuresets = {
         # the train, and test vectors must be supplied here.

        'termFrequency unigram': [UnigramVectorizer.fit_transform(train_X).toarray(), UnigramVectorizer.transform(test_X).toarray()],
        'termFrequency bigrams': [BigramVectorizer.fit_transform(train_X).toarray(), BigramVectorizer.transform(test_X).toarray()],
        'word2vec 300 vectorsize - avg':[w2v_train_X, w2v_test_X],
        #'word2vec 300 vectorsize - tfIdf weighted avg':[w2v_train_X, w2v_test_X]

    }


    classalgs = {'Random': algs.Classifier(), 
                 'Naive Bayes': algs.NaiveBayesMN(),
                 #'Ridge Linear Regression': algs.LinearRegressionClass(),
                 #'Logistic Regression': algs.LogitReg()
                 
                 #'L1 Logistic Regression': algs.LogitReg({'regularizer': 'l1'}),
                 #'L2 Logistic Regression': algs.LogitReg({'regularizer': 'l2'}),
                 #'Logistic Alternative': algs.LogitRegAlternative(),                 
                 #'Neural Network': algs.NeuralNet({'epochs': 100})
                }  
    numalgs = len(classalgs)    

    
        
    errors = {}
    for featuresetname in featuresets:
         for learnername in classalgs:
             errors['_'.join([featuresetname, learnername])] = float(0)
                


    

    print('Running on train={0} and test={1} samples for run').format(train_X.shape, test_X.shape)

        
    for featuresetname, featureset in featuresets.iteritems():
    	
    	print "Feature set name: ", featuresetname

        Xtrain = featureset[0]
        Xtest = featureset[1]
        for learnername, learner in classalgs.iteritems():
        
            # Reset learner for new parameters
    	    print '    Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
    	    # Train model
    	    learner.learn(Xtrain = Xtrain, ytrain = train_y)
    	    # Test model
    	    predictions = learner.predict(Xtest = Xtest)
    	    error = utils.geterror(test_y, predictions)
    	    print '        Error for ' + learnername + ': ' + str(error)
            errors['_'.join([featuresetname, learnername])] = error

 
    





    