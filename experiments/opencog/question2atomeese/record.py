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
    
    @staticmethod
    def fromString(string):
        record = Record()
        (record.questionId, record.questionType, 
         record.question, record.imageId, record.answer,
         record.formula, record.groundedFormula) = string.split('::')
        return record
