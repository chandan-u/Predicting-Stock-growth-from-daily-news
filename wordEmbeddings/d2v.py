from gensim import models

import sys
sys.path.append('../')
import dataloader as dtl

# class LabeledLineSentence(object):
#     def __init__(self, filename):
#         self.filename = filename
#     def __iter__(self):
#         for uid, line in enumerate(open(filename)):
#             yield LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])

train_X, train_y, test_X, test_y  = dtl.loadDataCombinedColumns("../data/Combined_News_DJIA.csv")



sentences = []
index = 0
for sentence, label in zip(train_X, train_y):
    index = index+1
    sentence = models.doc2vec.LabeledSentence( words= sentence.split(), tags=[indexlabel])
    sentences.append(sentence)


            
model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
model.build_vocab(sentences)

for epoch in range(10):
    model.train(sentences, total_examples = len(sentences), epochs= 10)
    model.alpha -= 0.002  # decrease the learning rate`
    model.min_alpha = model.alpha  # fix the learning rate, no decay

model.save("my_model.doc2vec")

model_loaded = models.Doc2Vec.load('my_model.doc2vec')

print model.docvecs.most_similar(["github"])
print model_loaded.docvecs.most_similar(["crude"])