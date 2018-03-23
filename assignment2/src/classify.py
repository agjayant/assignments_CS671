import numpy as np
from feat import bbow, tfR, tfIdf, wv
from algo import useSVM, useNB, useLR, useMLP
import argparse

import warnings
warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser(description="Sentiment Analysis")
argparser.add_argument('-f', '--feat', help="Representation", default="bbow")
argparser.add_argument('-c', '--classify', help="Classification Algorithm", default="svm")
argparser.add_argument('-s', '--samples', help="Number of Training Samples(upto 12500)", default=1000)

def getVector(vec, size=89527):

    bbow_v = np.zeros(size)
    bbow_v[vec] = 1
    return bbow_v

if __name__ == "__main__":

    args = argparser.parse_args()

    num_examples = int(args.samples)

    if args.feat == "bbow":
        train_data, train_y, test_data, test_y = bbow(num_examples)
        train_x = [getVector(vec) for vec in train_data]
        test_x = [getVector(vec) for vec in test_data]
    elif args.feat == "tf":
        train_x, train_y, test_x, test_y = tfR(num_examples)
    elif args.feat == "tfidf":
        train_x, train_y, test_x, test_y = tfIdf(num_examples)
    elif args.feat == "wv":
        train_x, train_y, test_x, test_y = wv(num_examples)

    if args.classify == "svm":
        predictions = useSVM(train_x, train_y, test_x)
    elif args.classify == "nb":
        predictions = useNB(train_x, train_y, test_x)
    elif args.classify == "lr":
        predictions = useLR(train_x, train_y, test_x)
    elif args.classify == "mlp":
        predictions = useMLP(train_x, train_y, test_x)

    acc = sum([1 for i,j in zip(predictions, test_y) if i == j])
    print acc*0.5/num_examples

    # permList = np.random.permutation(range(len(train_data)))
    # train_data = train_data[permList]
    # train_label = train_label[permList]

    # permList = np.random.permutation(range(len(test_data)))
    # test_data = test_data[permList]
    # test_label = test_label[permList]





