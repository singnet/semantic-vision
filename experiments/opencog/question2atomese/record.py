import re

class Record:
    
    def __init__(self):
        self.question = None
        self.questionType = None
        self.questionId = None
        self.imageId = None
        self.answer = None
        self.formula = None
        self.groundedFormula = None
    
    def toString(self):
        return '{}::{}::{}::{}::{}::{}::{}'.format(self.questionId, 
                                                   self.questionType, 
                                                   self.question, self.imageId, 
                                                   self.answer, self.formula, 
                                                   self.groundedFormula);
    
    @classmethod
    def fromString(cls, string):
        record = cls()
        (record.questionId, record.questionType, 
         record.question, record.imageId, record.answer,
         record.formula, record.groundedFormula) = string.split('::')
        return record
    
    @classmethod
    def fromOther(cls, other):
        record = cls()
        record.question = other.question
        record.questionType = other.questionType
        record.questionId = other.questionId
        record.imageId = other.imageId
        record.answer = other.answer
        record.formula = other.formula
        record.groundedFormula = other.groundedFormula
        return record
    
    def getWords(self):
        # parse '_test(A, B);next(B, A)'
        words = re.split('^[^\(]+\(|\)[^\(]+\(|, |\)[^\(]*$', 
                         self.groundedFormula)
        return filter(lambda x: len(x) > 0, map(str.strip, words))
