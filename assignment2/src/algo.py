import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix

def useSVM(train_x, train_y, test_x):

    print "Training Beginning..."
    clf = LinearSVC()
    train_x = csr_matrix(train_x)
    clf.fit(train_x, train_y)
    print "Training Done..."

    predictions = [clf.predict(test_vec) for test_vec in test_x]

    return predictions

def useNB(train_x, train_y, test_x):

    print "Training Beginning..."
    clf = GaussianNB()
    # train_x = csr_matrix(train_x)
    clf.fit(train_x, train_y)
    print "Training Done..."

    predictions = [clf.predict(test_vec) for test_vec in test_x]

    return predictions

def useLR(train_x, train_y, test_x):

    print "Training Beginning..."
    clf = LogisticRegression()
    train_x = csr_matrix(train_x)
    clf.fit(train_x, train_y)
    print "Training Done..."

    predictions = [clf.predict(test_vec) for test_vec in test_x]

    return predictions

def useMLP(train_x, train_y, test_x):

    print "Training Beginning..."
    clf = MLPClassifier()
    train_x = csr_matrix(train_x)
    clf.fit(train_x, train_y)
    print "Training Done..."

    predictions = [clf.predict(test_vec) for test_vec in test_x]

    return predictions

