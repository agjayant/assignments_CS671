import re
import sys
import numpy as np
from sklearn import svm

textFile = open(sys.argv[1], 'r')
text_test = textFile.read()
textFile.close()

textFile_test = open(sys.argv[2], 'r')
testText = textFile_test.read()
textFile_test.close()

pat = re.compile(r'[a-z]([\.!?])[ "\n]+[A-Z]', re.M)

sentences = pat.finditer(text_test)
sentences1 = pat.finditer(testText)

indices = []
indices1 = []

for sentence in sentences:
    indices.append(sentence.start()+1)

for sentence in sentences1:
    indices1.append(sentence.start()+1)

len_train = len(text_test)
len_test = len(testText)

train_x = []
train_y = []

puncList = ['.','?', '!']

for i in range(len_train-1):

    if text_test[i] in puncList:

        ### Binary Vector
        this_vector = np.zeros(3)

        this_vector[2] = int(text_test[i+1] == ' ')

        if i+2 > len_train -1:
            this_vector[0] = 1
            this_vector[1] = 1
        else:

            this_vector[0] = int(text_test[i+2] >= 'A' and text_test[i+2] <= 'Z')
            this_vector[1] = int(text_test[i-1] >= 'a' and text_test[i-1] <= 'z')

        train_x.append(this_vector)

        if i in indices:
            train_y.append(1)
        else:
            train_y.append(0)

test_x = []
test_y= []

for i in range(len_test-1):

    if testText[i] in puncList:

        ### Binary Vector
        this_vector = np.zeros(3)

        this_vector[2] = int(testText[i+1] == ' ')

        if i+2 > len_test -1 :
            this_vector[0] = 1
            this_vector[1] = 1
        else:
            this_vector[0] = int(testText[i+2] >= 'A' and testText[i+2] <= 'Z')
            this_vector[1] = int(testText[i-1] >= 'a' and testText[i-1] <= 'z')

        test_x.append(this_vector)

        if i in indices1:
            test_y.append(1)
        else:
            test_y.append(0)

clf = svm.SVC()
clf.fit(train_x, train_y)

acc = 0
for i in range(len(test_x)-1):
    pred =  clf.predict([test_x[i]])
    acc += pred[0] == test_y[i]

print acc*100.0/len(test_x)
