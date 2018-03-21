import numpy as np
from feat import bbow
from sklearn import svm

def getVector(vec, size=89527):

    bbow_v = np.zeros(size)
    bbow_v[vec] = 1
    return bbow_v

if __name__ == "__main__":
    num_examples = 200 ## upto 12500
    train_data, train_label, test_data, test_label = bbow(num_examples)

    # permList = np.random.permutation(range(len(train_data)))
    # train_data = train_data[permList]
    # train_label = train_label[permList]

    # permList = np.random.permutation(range(len(test_data)))
    # test_data = test_data[permList]
    # test_label = test_label[permList]

    train_x = [getVector(vec) for vec in train_data]
    test_x = [getVector(vec) for vec in test_data]

    print "Training Beginning..."
    clf = svm.SVC()
    clf.fit(train_x, train_label)
    print "Traning Done..."

    predictions = [clf.predict(test_vec) for test_vec in test_x]
    acc = sum([1 for i,j in zip(predictions, test_label) if i == j])
    print acc*0.5/num_examples




