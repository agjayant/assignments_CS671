import re
import sys

textFile = open(sys.argv[1], 'r')
text_test = textFile.read()
textFile.close()

pat = re.compile(r'[a-z]([\.!?])[ "\n]+[A-Z]', re.M)

sentences = pat.finditer(text_test)

indices = [0]

for sentence in sentences:
    indices.append(sentence.start()+2)

parts = [text_test[i:j] for i,j in zip(indices, indices[1:]+[None])]

newFile = open(sys.argv[2],'w')
for s in parts:
    newFile.write('<s>')
    newFile.write(s)
    newFile.write('<\s>')
newFile.close()


