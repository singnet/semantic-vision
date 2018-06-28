#!/usr/bin/python3
import ijson
import logging
import argparse

class Record:

    def __init__(self):
        self.question = None
        self.questionType = None
        self.questionId = None
        self.imageId = None
        self.answer = None

    def toString(self):
        return '{}::{}::{}::{}::{}'.format(self.questionId, self.questionType, self.question, self.imageId, self.answer);

def getMostFrequentAnswer(answers):
    maxCount = 0
    maxAnswer = None
    for answer, count in answers.items():
        if count > maxCount:
            maxCount = count
            maxAnswer = answer
    return maxAnswer

parser = argparse.ArgumentParser(description='Convert set of questions and '
                                 'annotations to plain file with delimiters.')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test mode, process only 10 first questions')
parser.add_argument('--loglevel', dest='loggingLevel', action='store',
                    type = str, default = 'INFO',
                    help='logging level: DEBUG, INFO, ERROR')
args = parser.parse_args()

log = logging.getLogger("get_questions")
log.setLevel(args.loggingLevel)
log.addHandler(logging.StreamHandler())

parser = ijson.parse(open('v2_OpenEnded_mscoco_train2014_questions.json'))

records = {}
record = None

i = 0
for prefix, event, value in parser:
    if args.test and i > 10:
        break

    log.debug('%s, %s, %s', prefix, event, value)

    if (prefix == 'questions.item.image_id'):
        record = Record()
        record.imageId = value

    if (prefix == 'questions.item.question'):
        record.question = value

    if (prefix == 'questions.item.question_id'):
        record.questionId = value
        records[record.questionId] = record
        i += 1

parser = ijson.parse(open('v2_mscoco_train2014_annotations.json'))

i = 0
answerType = None
answers = {}
for prefix, event, value in parser:
    if args.test and i > 10:
        break

    log.debug('%s, %s, %s', prefix, event, value)

    if (prefix == 'annotations.item.answers.item.answer'):
        count = answers.setdefault(value, 0) + 1
        answers[value] = count

    if (prefix == 'annotations.item.answer_type'):
        answerType = value

    if (prefix == 'annotations.item.question_id'):
        records[value].questionType = answerType
        records[value].answer = getMostFrequentAnswer(answers)
        answers = {}
        i += 1

for questionId, record in records.items():
        print(record.toString())
