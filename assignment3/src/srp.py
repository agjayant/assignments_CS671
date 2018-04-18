from conllu import parse, parse_tree, print_tree
from gensim.models import KeyedVectors
import tqdm
import numpy as np
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")

N_GLOVE_TOKENS = 400000

fTrain =  "../UD_English-EWT/en_ewt-ud-train.conllu"

with open(fTrain) as f:
    train_raw = f.readlines()

posFile = 'posTags.txt'

with open(posFile) as f:
    postags= f.readlines()

posDict = {}
ind = 0
for line in postags:
    posDict[line[:-1]] = ind
    ind += 1

cur_pointer = 3

train_x = []
train_y = []

test_x = []
test_y = []

num_examples = 1000
num_test_examples = 100
count = 0
# model_name = '../../assignment2/GoogleNews-vectors-negative300.bin'
# model = KeyedVectors.load_word2vec_format(model_name, binary=True)

model = dict()
with open("../../assignment2/glove.6B.50d.txt") as f:
        for line in tqdm.tqdm(f, total=N_GLOVE_TOKENS):
                values = line.split()
                word, coefficients = values[0], np.asarray(values[1:], dtype=np.float32)
                model[word] = coefficients

def get_word(word, dim, word_dict):

    try:
        word_feat = model[word_dict[word][0]]
    except:
        word_feat = np.zeros(dim)

    pos_feat = np.zeros(17)
    try:
        pos_feat[word_dict[word][1]] = 1.0
    except:
        pass

    return np.concatenate((word_feat, pos_feat))

def get_feature(config, word_dict):

    dim1 = 50
    dim = dim1 + 17   ## postag
    num_context = 3

    feat = np.zeros(dim*2*num_context)
    i = 0
    stack_len = len(config[0])
    buf_len = len(config[1])

    while i < min(num_context, stack_len):
        feat[i*dim:(i+1)*dim] = get_word(config[0][-1-i], dim1, word_dict)
        i += 1

    i = 0
    while i < min(num_context, buf_len):
        feat[(i+num_context)*dim:(i+num_context+1)*dim] = get_word(config[1][-1-i], dim1, word_dict)
        i += 1

    return feat

def next_sen():

    global cur_pointer

    next_raw = train_raw[cur_pointer].split(' ')
    is_new = next_raw[0] == '#' and next_raw[1] == 'text'

    sen_data = ''
    while not is_new :
        sen_data += train_raw[cur_pointer]
        cur_pointer += 1

        next_raw = train_raw[cur_pointer].split(' ')
        is_new = next_raw[0] == '#' #and next_raw[1] == 'text'

    next_raw = train_raw[cur_pointer].split('\t')
    is_new = next_raw[0] == '1'

    while not is_new:
        cur_pointer += 1
        next_raw = train_raw[cur_pointer].split('\t')
        is_new = next_raw[0] == '1'

    return sen_data

def add_to_training(sentence, test=False, debug=False):

    global train_x
    global test_x
    global train_y
    global test_y
    global count

    try:
        sen_parse = parse(sentence)
    except:
        return
    # sen_tree = parse_tree(sentence)

    ## Get configurations and operations
    num_words = len(sen_parse[0])
    head_dict = {0:0}
    child_dict = {}
    word_dict = {}
    for word in range(num_words+1):
        child_dict[word] = []

    for word in sen_parse[0]:
        try:
            head_dict[int(word['id'])] = int(word['head'])
        except:
            return
        child_dict[int(word['head'])].append(int(word['id']))
        word_dict[int(word['id'])] = (word['lemma'], posDict[word['upostag']])

    configs = []
    operations = []
    cur_config = [[0],list(reversed(range(1,num_words+1)))]

    while (len(cur_config[1]) !=0 ):

        if len(cur_config[0]) == 0: break

        temp = cur_config[:]
        configs.append(temp)
        if debug: print cur_config,

        if head_dict[cur_config[0][-1]] == cur_config[1][-1] :
            if (len(child_dict[cur_config[0][-1]]) == 0) or (max(child_dict[cur_config[0][-1]]) < cur_config[0][-1]):
                cur_config[0] = cur_config[0][:-1]
                # LEFT
                operations.append(0)
            else:
                cur_config[0].append(cur_config[1][-1])
                cur_config[1] = cur_config[1][:-1]
                operations.append(2)

        elif head_dict[cur_config[1][-1]] == cur_config[0][-1] :
            if (len(child_dict[cur_config[1][-1]]) == 0) or (max(child_dict[cur_config[1][-1]]) < cur_config[1][-1]):
                # RIGHT
                operations.append(1)
                cur_config[1] = cur_config[1][:-1]
                cur_config[1].append(cur_config[0][-1])
                cur_config[0] = cur_config[0][:-1]
            else:
                cur_config[0].append(cur_config[1][-1])
                cur_config[1] = cur_config[1][:-1]
                operations.append(2)
        else:
            ## SHIFT
            operations.append(2)
            cur_config[0].append(cur_config[1][-1])
            cur_config[1] = cur_config[1][:-1]

        if debug: print operations[-1]

    if debug: print cur_config

    if test:
        for conf in range(len(configs)):
            test_x.append(get_feature(configs[conf], word_dict))
            test_y.append(operations[conf])

            count += 1

    else:
        for conf in range(len(configs)):
            train_x.append(get_feature(configs[conf], word_dict))
            train_y.append(operations[conf])

            count += 1

for ex in range(num_examples):
    add_to_training(next_sen())

print "Train Examples ", count
count = 0

for ex in range(num_test_examples):
    add_to_training(next_sen(), test=True)

print "Test Examples ", count

print "Training Begin..."
clf = MLPClassifier()
clf.fit(train_x, train_y)
print "Training Done."

print clf.score(test_x, test_y)





