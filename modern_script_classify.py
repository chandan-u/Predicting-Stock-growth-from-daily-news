#from __future__ import division  # floating point division

import math
import numpy as np
import itertools
from pyspark import SparkContext as sc


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


    # Algorithm Name, Algorithm class, parameter combinations to algorithm(list of lists), featuresets to experimented(list of featureset)
    
    pipeline = {

        "Random": [algs.Classifier(),None,None],
        "Multinomial Naive Bayes": [algs.NaiveBayesMN, None, ['termFrequency unigram','termFrequency bigrams', 'word2vec 300 vectorsize - avg' ]]
    }


    numalgs = len(pipeline)    

    ## initialize error    
    errors = {}
    for learnername in pipeline:
        
        parameters_list = pipeline[learnername][1]
        featureset_list = pipeline[learnername][2]

        if learnername is "Random":
            errors['_'.join([learnername,'',''])] = float(0)
            continue

        if parameters_list is None:
            for featureset in featureset_list:
                errors['_'.join([learnername,'',featuresetname])] = float(0)

        else:
            if featureset_list is None:
                raise Exception("No features are provided for "+learnername)
            for parametername in parameters_list:
                for featureset in featureset_list:
                    errors['_'.join([learnername, parametername,featuresetname])] = float(0)

                 


    print('Running on train={0} and test={1} samples for run').format(train_X.shape, test_X.shape)

    


    def run_experiment(learner, parameters, featuresetname):


        #learnername, parameters, featuresetname,
        Xtrain, Xtest = featuresets[featuresetname]
        learner.learn(Xtrain = Xtrain, ytrain = train_y)
        # Test model
        predictions = learner.predict(Xtest = Xtest)
        error = utils.geterror(test_y, predictions)
        print '        Error for ' + learnername +  parameters + featuresetname + ': ' + str(error)
        errors['_'.join([featuresetname, parameters,learnername])] = error

    for learnername in pipeline:

        print '    Running learner = ' + learnername

        learner =  pipeline[learnername][0]
        
        
        parameters_list = pipeline[learnername][1]
        featureset_list = pipeline[learnername][2]
        if parameters_list is None:
            parameters_list = ['']
        if featureset_list is None:
            featureset_list = ['']

        
        experiments = [ experiment for experiment in itertools.product(parameters_list, featureset_list)]


        experiments_rdd = sc.parallelize(experiments)


        experiments_rdd.map(lambda parameters, featuresetname : run_experiment(learner, parameters, featuresetname))
            
        print '    Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()) + 'on features ' + featuresetname

        
        learner.reset(parameters)






            
        


    
    


   
    





    