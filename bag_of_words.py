from data import loadDataCombinedColumns
from sklearn.feature_extraction.text import CountVectorizer





train_X, train_y, test_X, test_y  = loadDataCombinedColumns()

print "train shape, test shape, type of dataframe : ", train_X.shape, test_y.shape, type(train_X)
 
# Create an vectorizer instance : unigram, stopwords removed, lowercase=True
UnigramVectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1), stop_words='english')


# fit: learns the vocabulary, transform: returns the document-term matrix
train_unigram_matrix = UnigramVectorizer.fit_transform(train_X)


# The unigram matrix is of type scipy.sparse.csr.csr_matrix: 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
print "train Unigram matrix shape:, ", train_unigram_matrix.shape, " type: ", type(train_unigram_matrix)
print train_unigram_matrix[1]



test_unigram_matrix  = UnigramVectorizer.transform(test_X)

print "test_unigram_matrix  shape:, ", test_unigram_matrix.shape, " type: ", type(test_unigram_matrix)
print test_unigram_matrix[1]

print train_unigram_matrix.toarray().shape



































