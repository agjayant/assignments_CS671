import numpy as np
import sys

word_to_idf = np.zeros(89527)

def getBBOW(line):

    review_words = []
    temp = line[:-1].split(' ')
    for word in temp[1:]:
        review_words.append(int(word.split(':')[0]))

    return review_words

def getTFR(line):

    tf_rep = np.zeros(89527, "float32")
    temp = line[:-1].split(' ')
    total_words = 0
    for word in temp[1:]:
        temp2 = word.split(':')
        tf_rep[int(temp2[0])] = int(temp2[1])
        word_to_idf[int(temp2[0])] += 1
        total_words += int(temp2[1])

    return tf_rep*10.0/total_words

def bbow(num_ex):

    trainFile = '../aclImdb/train/labeledBow.feat'
    with open(trainFile) as f:
        trainData = f.readlines()

    train_bbow = []
    train_label = []
    for i in range(num_ex) :

        train_bbow.append(getBBOW(trainData[i]))
        train_label.append(1)

        j = i + 12500
        train_bbow.append(getBBOW(trainData[j]))
        train_label.append(0)

    testFile = '../aclImdb/test/labeledBow.feat'
    with open(testFile) as f:
        testData = f.readlines()

    test_bbow = []
    test_label = []
    for i in range(num_ex):

        test_bbow.append(getBBOW(testData[i]))
        test_label.append(1)

        j = i+12500
        test_bbow.append(getBBOW(testData[j]))
        test_label.append(0)

    return train_bbow, train_label, test_bbow, test_label

def tfR(num_ex):

    trainFile = '../aclImdb/train/labeledBow.feat'
    with open(trainFile) as f:
        trainData = f.readlines()

    train_bbow = []
    train_label = []
    for i in range(num_ex) :

        train_bbow.append(getTFR(trainData[i]))
        train_label.append(1)

        j = i + 12500
        train_bbow.append(getTFR(trainData[j]))
        train_label.append(0)

    testFile = '../aclImdb/test/labeledBow.feat'
    with open(testFile) as f:
        testData = f.readlines()

    test_bbow = []
    test_label = []
    for i in range(num_ex):

        test_bbow.append(getTFR(testData[i]))
        test_label.append(1)

        j = i+12500
        test_bbow.append(getTFR(testData[j]))
        test_label.append(0)

    return train_bbow, train_label, test_bbow, test_label

def tfIdf(num_ex):

    global word_to_idf

    trainFile = '../aclImdb/train/labeledBow.feat'
    with open(trainFile) as f:
        trainData = f.readlines()

    train_bbow = []
    train_label = []
    for i in range(num_ex):

        train_bbow.append(getTFR(trainData[i]))
        train_label.append(1)

        j = i + 12500
        train_bbow.append(getTFR(trainData[j]))
        train_label.append(0)

    train_bbow = np.asarray(train_bbow)

    for i in range(89527):
        if word_to_idf[i] > 0:
            train_bbow[:,i] *= np.log(2.0*num_ex/word_to_idf[i])

    word_to_idf = np.zeros(89527)

    testFile = '../aclImdb/test/labeledBow.feat'
    with open(testFile) as f:
        testData = f.readlines()

    test_bbow = []
    test_label = []
    for i in range(num_ex):

        test_bbow.append(getTFR(testData[i]))
        test_label.append(1)

        j = i+12500
        test_bbow.append(getTFR(testData[j]))
        test_label.append(0)

    test_bbow = np.asarray(test_bbow)

    for i in range(89527):
        if word_to_idf[i] > 0:
            test_bbow[:,i] *= np.log(2.0*num_ex/word_to_idf[i])


    return train_bbow, train_label, test_bbow, test_label





