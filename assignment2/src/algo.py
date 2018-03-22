import numpy as np
from sklearn import svm

def useSVM(train_x, train_y, test_x):

    print "Training Beginning..."
    clf = svm.SVC()
    clf.fit(train_x, train_y)
    print "Traning Done..."

    predictions = [clf.predict(test_vec) for test_vec in test_x]

    return predictions

