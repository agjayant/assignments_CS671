import re
import sys

textFile = open(sys.argv[1], 'r')
text_test = textFile.read()
textFile.close()

sone = re.sub("'(?=[stvem][ ,.\n])", "$", text_test)
stwo =  re.sub("s' ", "s$ ", sone)
sthree = re.sub("'", "\"", stwo)
sfour = re.sub("\$", "'", sthree)

newFile = open(sys.argv[2], 'w')
newFile.write(sfour)
newFile.close()

