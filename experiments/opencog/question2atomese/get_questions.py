#!/usr/bin/python3

import ijson
import logging
import argparse
from record import Record

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
parser.add_argument('--questions', '-q', dest='questionsFileName',
                    action='store', type=str, required=True,
                    help='questions json filename')
parser.add_argument('--annotations', '-a', dest='annotationsFileName',
                    action='store', type=str, default=None,
                    help='annotations json filename')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test mode, process only 10 first questions')
parser.add_argument('--loglevel', dest='loggingLevel', action='store',
                    type = str, default='INFO',
                    choices=['INFO', 'DEBUG', 'ERROR'], 
                    help='logging level')
args = parser.parse_args()

log = logging.getLogger("get_questions")
log.setLevel(args.loggingLevel)
log.addHandler(logging.StreamHandler())

parser = ijson.parse(open(args.questionsFileName))

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

if args.annotationsFileName != None:
    parser = ijson.parse(open(args.annotationsFileName))
    
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
