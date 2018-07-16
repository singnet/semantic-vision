#!/usr/bin/python3

import argparse
import sys

from record import Record

parser = argparse.ArgumentParser(description='Get words from file')
parser.add_argument('--mode', dest='mode', action='store',
                    type = str, default='WORDS',
                    choices=['WORDS', 'STATISTICS'], 
                    help='output mode')
args = parser.parse_args()

class Word:
    
    def __init__(self, word):
        self.word = word
        self.records = set()
        
    def addRecord(self, record):
        self.records.add(record)
        
    def count(self):
        return len(self.records)

words = {}

for line in sys.stdin:
    record = Record.fromString(line)
    for word in record.getWords():
        words.setdefault(word.lower(), Word(word.lower())).addRecord(record)

if args.mode == 'WORDS':
    for word in words.keys():
        print(word)
elif args.mode == 'STATISTICS':
    sortedWords = sorted(words.keys(), key=lambda word: words[word].count())
    for word in sortedWords:
        print('Word: {}, count: {}'.format(word, words[word].count()))
else:
    raise AttributeError('Incorrect output mode: {}'.format(args.mode))