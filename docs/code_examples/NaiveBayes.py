# def naiveBayes(train, test):

#     """
#        naiveBayes: To classify Djia Stock average increased (1) or decreased (0).
#        Just on unigrams only. 
#        If a word/feature from test set is not found in train: these words are skipped.
#        for both 0 and 1 cases. More at https://www.youtube.com/watch?v=EqjyLfpv5oA
#     """
    
#     # Generative model: requires prob cal for each case
#     train_0 = train[train['Label'] == 0]
#     train_1 = train[train['Label'] == 1]
    
#     # get grams/tokens
#     unigrams_0, bigrams, trigrams = getGramsDF(train_0)
#     unigrams_1, bigrams, trigrams = getGramsDF(train_1)
#     #unigrams_test, bigrams, trigrams = getGrams(test)
#     # get freq counts for each language feature from train set
#     # in this prob: only unigrams counts are computed
#     train_0_dict = getUnigramFreq(unigrams_0)
#     train_1_dict = getUnigramFreq(unigrams_1)

    
#     # lengths
#     train_0_len = train_0.shape[0]
#     train_1_len = train_1.shape[0]
#     train_len = train.shape[0]


#     #prior prob
#     prior_0 = float(train_0_len)/train_len
#     prior_1 = float(train_1_len)/train_len

#     pred = []
#     for index, row in test.iloc[:,1:25].iterrows():
        
#         features = []

#         # preprocess test sent: extract features
#         map(lambda sentence:features.extend(getGrams(sentence)), row)

#         # filter words that are not present in train (we skip these while calc prob)
#         # entire row is considered one sentence:
        
#         #words_0 = filter(lambda word:  word in unigrams_0 , words)
#         #words_1 = filter(lambda word:  word in unigrams_1 , words)     
#         #prob_0 =   reduce(lambda x, word: x * train_0_dict[word]/float(train_0_len), words_0, prior_0)
#         #prob_1 =   reduce(lambda x, word: x * train_1_dict[word]/float(train_1_len), words_1, prior_1)
 
#         prob_0 = prior_0
#         prob_1 = prior_1
        
#         for feature in features:
#             if feature in train_0_dict.keys():
#                 prob_0 = prob_0 * long(train_0_dict[feature])/long(train_0_len)
#             if feature in train_1_dict.keys():
#                 print train_1_dict[feature]
#                 prob_1 = prob_1 * long(train_1_dict[feature])/long(train_1_len)
#         print prob_0, prob_1
#         predicted = 0 if prob_0 > prob_1  else 1
#         pred.append(predicted)  

#     # Accuracy    
#     accuracy(pred, test['Label'])

          


