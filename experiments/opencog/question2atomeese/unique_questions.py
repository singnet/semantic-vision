#!/usr/bin/python3

import argparse
import re

from record import Record


def loadWordsFromFile(fileName):
    words = set()
    file = open(fileName, 'r')
    for line in file:
        record = Record.fromString(line)
        for word in getWords(record.groundedFormula):
            words.add(word)
    return words

parser = argparse.ArgumentParser(description='Calculate number of questions '
                                 'containing unque words')
parser.add_argument('--train', '-q', dest='trainFileName',
                    action='store', type=str, required=True,
                    help='train questions filename')
parser.add_argument('--test', '-a', dest='testFileName',
                    action='store', type=str, default=None,
                    help='test questions filename')
args = parser.parse_args()

def getWords(groundedFormula):
    # parse '_test(A, B);next(B, A)'
    words = re.split('\)[^\(]+\(|, |^[^\(]+\(|\)[^\(]+$', groundedFormula)
    return map(str.strip, words)

trainWords = loadWordsFromFile(args.trainFileName)
testWords = loadWordsFromFile(args.testFileName)
uniqueWords = testWords - trainWords

print('Number of words:')
print('train set - ', len(trainWords))
print('test set - ', len(testWords))
print('unique words in test set - ', len(uniqueWords))

file = open(args.testFileName, 'r')
for line in file:
    record = Record.fromString(line)
    for word in getWords(record.groundedFormula):
        if word in uniqueWords:
            print('{} (unique word: {})'.format(record.question, word))
            break
