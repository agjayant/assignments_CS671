from conllu import parse, parse_tree, print_tree

fTrain =  "../UD_English-EWT/en_ewt-ud-train.conllu"

with open(fTrain) as f:
    train_raw = f.readlines()

cur_pointer = 3

pos_list = []

def next_sen():

    global cur_pointer

    next_raw = train_raw[cur_pointer].split(' ')
    is_new = next_raw[0] == '#' and next_raw[1] == 'text'

    sen_data = ''
    while not is_new :
        sen_data += train_raw[cur_pointer]
        cur_pointer += 1

        next_raw = train_raw[cur_pointer].split(' ')
        is_new = next_raw[0] == '#' and next_raw[1] == 'text'

    cur_pointer += 1

    return sen_data

def add_to_training(sentence, test=False, debug=False):

    global pos_list

    try:
        sen_parse = parse(sentence)
    except:
        return
    # sen_tree = parse_tree(sentence)

    ## Get configurations and operations
    num_words = len(sen_parse[0])

    for word in sen_parse[0]:
        if word['upostag'] not in pos_list:
            pos_list.append(word['upostag'])

for i in range(1000):
    add_to_training(next_sen())

posF = open('posTags.txt', 'w')
for pos in pos_list:
    posF.write(pos)
    posF.write('\n')
