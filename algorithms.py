

from __future__ import division  # Force all divisions to be floating point division
import numpy as np
import random
from nltk.tokenize import word_tokenize
import tensorflow as tf


import utilities as utils
import dataloader as dtl




class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest





class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        # default and fixed list of params
        self.params = {'regwgt': 0.01}
        # Update params
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest


class NaiveBayesMN(Classifier):
    """
       Multinomial Naive Bayes:
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)
        
    
    def learn(self, Xtrain, ytrain):
        """
            designed for binary classification only
        """
    
        # Generative model: requires prob cal for each case
        self.train_0 = Xtrain[ytrain == 0]             
        self.train_1 = Xtrain[ytrain == 1]
    
    
    
        # sizes
        train_0_len = self.train_0.shape[0]
        train_1_len = self.train_1.shape[0]
        train_len = Xtrain.shape[0]
        self.class0_size = np.sum(self.train_0)        # total words/count in class 0
        self.class1_size = np.sum(self.train_1)        # total words/count in class 1
        self.V = Xtrain.shape[1]    # Vocabulary size

        # prior prob
        self.prior_0 = (train_0_len)/train_len    # prior for class 0
        self.prior_1 = (train_1_len)/train_len    # prior for class 1


        
    def predict(self, Xtest):
        # Xtest is a series
        
        pred = []
        for row in Xtest:

            prob_0 = self.prior_0
            prob_1 = self.prior_1
        
            # info: https://www.youtube.com/watch?v=pc36aYTP44o
            # this formula for p(feature/class) is called m-estimate
            for index, feature in enumerate(row):
                # remove the if condition to multiply prob for all attributes.
                if feature != 0:
                    prob_0 = prob_0 * (np.sum(self.train_0[:,index])+1)/(self.class0_size+self.V)
                    prob_1 = prob_1 * (np.sum(self.train_1[:,index])+1)/(self.class1_size+self.V)
            predicted = 0 if prob_0 > prob_1  else 1
            pred.append(predicted)  
        return pred







class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        # default and final list of params
        # only the ones list in self.params can be supplied
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest


class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions 

    def probabilityOfOne(self, weights, Xtrain):

        return 1/( 1 + np.exp(np.dot(weights.T, Xtrain))) 


    def learn(self, Xtrain, ytrain):
       """ Learns using the traindata """

       # Initial random weights ( Better if initialized using linear regression optimal wieghts)
       #Xless = Xtrain[:,self.params['features']]
       weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),ytrain)



       # w(t+1) = w(t) + eta * v
       #pone = self.probabilityOfOne(self.weights, Xtrain[i])
       p = utils.sigmoid(np.dot(Xtrain, weights))
       tolerance = 0.1
       #error = utils.crossentropy( Xtrain, ytrain, self.weights)
       error = np.linalg.norm(np.subtract(ytrain, p))
       err = np.linalg.norm(np.subtract(ytrain,  p))
       #err = 0
       #soldweights =self.weights
       while np.abs(error - err) < tolerance:
           P = np.diag(p)
           
           I = np.identity(P.shape[0])
           #Hess_inv =-np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,self.P),np.subtract(I,self.P)),Xtrain))
           #Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
           Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
           First_Grad= np.dot(Xtrain.T, np.subtract(ytrain,p))#np.dot(Xtrain.T, np.subtract(ytrain, p))
           #oldweights = self.weights
           weights= weights - (np.dot(Hess_inv, First_Grad))
           p = utils.sigmoid(np.dot(Xtrain, weights))

           # error = utils.crossentropy(Xtrain, ytrain, self.weights)
           err = np.linalg.norm(np.subtract(ytrain,  p))

       self.weights = weights

    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(ytest)
        return ytest




def ConnvolutionNeuralNetwork(Classifier):

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.resetparams(parameters)
            
    
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """


       #input_layer = tf.reshape(features, [-1, 28, 28, 1])


        
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest



    
        

