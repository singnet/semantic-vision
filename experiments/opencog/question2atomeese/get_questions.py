#!/usr/bin/python3
import ijson
import logging

class Record:

    def __init__(self):
        self.question = None
        self.answerType = None
        self.questionId = None
        self.imageId = None

    def toString(self):
        return '{}:{}:{}:{}'.format(self.questionId, self.answerType, self.question, self.imageId);

test = False

log = logging.getLogger("get_questions")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

parser = ijson.parse(open('v2_OpenEnded_mscoco_train2014_questions.json'))

records = {}
record = None

i = 0
for prefix, event, value in parser:
    if test and i > 10:
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
for prefix, event, value in parser:
    if test and i > 10:
        break

    log.debug('%s, %s, %s', prefix, event, value)

    if (prefix == 'annotations.item.answer_type'):
        answerType = value

    if (prefix == 'annotations.item.question_id'):
        records[value].answerType = answerType
        i += 1

for questionId, record in records.items():
        print(record.toString())
