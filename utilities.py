import pandas as pd
import numpy as np

"""

 Error Utilities: getConfusionMatrix/getaccuracy/geterror

"""

def getConfusionMatrix(pred, real):
    """
      Give accuracy, error rate, crossTable stats
    """
    # print pd.crosstab(pred, real)   
    
    total = float(real.shape[0])
    
    tp = 0   # true positive
    tn = 0   # true negitive
    fp = 0   # false positive
    fn = 0   # false negitive
    for predicted, actual in zip(pred, real):
        if predicted == actual:
            if predicted == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predicted == 1:
                fp += 1
            else:
                fn += 1
            

    print "(tp, tn, fp, fn):" , tp, tn, fp, fn
    print "accuracy is :", (tp+tn)/total


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0
    

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))







def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes


def update_dictionary_items(dict1, dict2):
    """ Replace any common dictionary items in dict1 with the values in dict2 
    There are more complicated and efficient ways to perform this task,
    but we will always have small dictionaries, so for our use case, this simple
    implementation is acceptable.
    """
    for k in dict1:
        if k in dict2:
            dict1[k]=dict2[k]



def l2(vec):
    """ l2 norm on a vector """
    return np.linalg.norm(vec)

def dl2(vec):
    """ Gradient of l2 norm on a vector """
    return vec

def l1(vec):
    """ l1 norm on a vector """
    return np.linalg.norm(vec, ord=1)

def dl1(vec):
    """ Subgradient of l1 norm on a vector """
    grad = np.sign(vec)
    grad[abs(vec) < 1e-4] = 0.0
    return grad



    