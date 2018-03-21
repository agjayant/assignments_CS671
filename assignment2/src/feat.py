import numpy as np
import sys
# sys.path.insert(0,'../aclImdb/')


def bbow(num_ex):

    trainFile = '../aclImdb/train/labeledBow.feat'
    with open(trainFile) as f:
        trainData = f.readlines()

    train_bbow = []
    train_label = []
    for i in range(num_ex) :

        review_words = []
        temp = trainData[i][:-1].split(' ')
        for word in temp[1:]:
            review_words.append(int(word.split(':')[0]))
        train_bbow.append(review_words)

        # train_label.append(int(i<12500))
        train_label.append(1)

        j = i + 12500
        review_words = []
        temp = trainData[j][:-1].split(' ')
        for word in temp[1:]:
            review_words.append(int(word.split(':')[0]))
        train_bbow.append(review_words)

        # train_label.append(int(j<12500))
        train_label.append(0)


    testFile = '../aclImdb/test/labeledBow.feat'
    with open(testFile) as f:
        testData = f.readlines()

    test_bbow = []
    test_label = []
    for i in range(num_ex):

        review_words = []
        temp = testData[i][:-1].split(' ')
        for word in temp[1:]:
            review_words.append(int(word.split(':')[0]))
        test_bbow.append(review_words)

        # test_label.append(int(i<12500))
        test_label.append(1)

        j = i+12500
        review_words = []
        temp = testData[j][:-1].split(' ')
        for word in temp[1:]:
            review_words.append(int(word.split(':')[0]))
        test_bbow.append(review_words)

        # test_label.append(int(i<12500))
        test_label.append(0)

    return train_bbow, train_label, test_bbow, test_label

if __name__== "__main__":
    train_x, train_y, test_x, test_y = bbow()


