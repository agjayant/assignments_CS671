import numpy as np
import sys
from gensim.models import KeyedVectors
import tqdm

N_GLOVE_TOKENS=400000

def computeIDF(trainData):

    word_to_idf = np.zeros(89527)

    for line in trainData:
        temp = line[:-1].split(' ')
        for word in temp[1:]:
            temp2 = word.split(':')
            word_to_idf[int(temp2[0])] += 1

    return word_to_idf

def getWV(model, word_list, line, dim, idf, word_to_idf):

    review_words = np.zeros(dim)
    temp = line[:-1].split(' ')
    total_words = 0
    for word in temp[1:]:
        word_ind = int(word.split(':')[0])
        count = int(word.split(':')[1])
        try:
            if idf and word_to_idf[word_ind] > 0:
                review_words += count*model[word_list[word_ind]]*np.log(25000/word_to_idf[word_ind])
            else:
                review_words += count*model[word_list[word_ind]]

            total_words += count
        except:
            continue

    return review_words/total_words


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


    trainFile = '../aclImdb/train/labeledBow.feat'
    with open(trainFile) as f:
        trainData = f.readlines()

    word_to_idf = computeIDF(trainData)

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
            train_bbow[:,i] *= np.log(25000/word_to_idf[i])

    testFile = '../aclImdb/test/labeledBow.feat'
    with open(testFile) as f:
        testData = f.readlines()

    word_to_idf = computeIDF(testData)

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
            test_bbow[:,i] *= np.log(25000/word_to_idf[i])


    return train_bbow, train_label, test_bbow, test_label


def wv(num_ex, idf=False):

    model_name = "../GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(model_name, binary=True)

    dim = 300

    word_index_file = open('../aclImdb/imdb.vocab')
    word_index_data = word_index_file.readlines()
    word_index = []
    for line in word_index_data:
        word_index.append(line[:-1])

    trainFile = '../aclImdb/train/labeledBow.feat'
    with open(trainFile) as f:
        trainData = f.readlines()

    word_to_idf = np.zeros(89527)
    if idf:
        word_to_idf = computeIDF(trainData)

    train_bbow = []
    train_label = []
    for i in range(num_ex):

        train_bbow.append(getWV(model, word_index, trainData[i], dim, idf, word_to_idf))
        train_label.append(1)

        j = i + 12500
        train_bbow.append(getWV(model, word_index, trainData[j], dim, idf, word_to_idf))
        train_label.append(0)

    train_bbow = np.asarray(train_bbow)

    testFile = '../aclImdb/test/labeledBow.feat'
    with open(testFile) as f:
        testData = f.readlines()

    if idf:
        word_to_idf = computeIDF(testData)

    test_bbow = []
    test_label = []
    for i in range(num_ex):

        test_bbow.append(getWV(model, word_index, testData[i], dim, idf, word_to_idf))
        test_label.append(1)

        j = i+12500
        test_bbow.append(getWV(model, word_index, testData[j], dim, idf, word_to_idf))
        test_label.append(0)

    test_bbow = np.asarray(test_bbow)

    return train_bbow, train_label, test_bbow, test_label

def wv_glove(num_ex, idf=False):

    model = dict()
    with open("../glove.6B.100d.txt") as f:
            for line in tqdm.tqdm(f, total=N_GLOVE_TOKENS):
                    values = line.split()
                    word, coefficients = values[0], np.asarray(values[1:], dtype=np.float32)
                    model[word] = coefficients

    dim = 100

    word_index_file = open('../aclImdb/imdb.vocab')
    word_index_data = word_index_file.readlines()
    word_index = []
    for line in word_index_data:
        word_index.append(line[:-1])

    trainFile = '../aclImdb/train/labeledBow.feat'
    with open(trainFile) as f:
        trainData = f.readlines()

    word_to_idf = np.zeros(89527)
    if idf:
        word_to_idf = computeIDF(trainData)

    train_bbow = []
    train_label = []
    for i in range(num_ex):

        train_bbow.append(getWV(model, word_index, trainData[i], dim, idf, word_to_idf))
        train_label.append(1)

        j = i + 12500
        train_bbow.append(getWV(model, word_index, trainData[j], dim, idf, word_to_idf))
        train_label.append(0)

    train_bbow = np.asarray(train_bbow)

    testFile = '../aclImdb/test/labeledBow.feat'
    with open(testFile) as f:
        testData = f.readlines()

    if idf:
        word_to_idf = computeIDF(testData)

    test_bbow = []
    test_label = []
    for i in range(num_ex):

        test_bbow.append(getWV(model, word_index, testData[i], dim, idf, word_to_idf))
        test_label.append(1)

        j = i+12500
        test_bbow.append(getWV(model, word_index, testData[j], dim, idf, word_to_idf))
        test_label.append(0)

    test_bbow = np.asarray(test_bbow)

    return train_bbow, train_label, test_bbow, test_label






