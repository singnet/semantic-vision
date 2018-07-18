import os
import logging
import jpype
import numpy as np
import datetime
import opencog.logger

from opencog.atomspace import AtomSpace, TruthValue, types
from opencog.type_constructors import *
from opencog.scheme_wrapper import scheme_eval, scheme_eval_v

import GetFeatures

# global variables
opencogLogLevel = 'NONE' # 'FINE'
pythonLogLevel = logging.ERROR # logging.DEBUG
currentDir = os.path.dirname(os.path.realpath(__file__))

log = None
questionConverter = None
atomspace = None


class Record:

    def __init__(self):
        self.question = None
        self.questionType = None
        self.questionId = None
        self.imageId = None

    def toString(self):
        return '{}:{}:{}:{}'.format(self.questionId, self.questionType,
                                    self.question, self.imageId);
        
    @staticmethod                            
    def fromString(line):
        record = Record()
        record.questionId, record.questionType, record.question, record.imageId = line.rstrip().split(":")
        return record


def getImageFileName(imageId):
    # TODO: implement
    # return '00000' + imageId + '.jpg'
    return 'hare.jpg'


def initializeAtomspace():
    atomspace = AtomSpace()
    set_type_ctor_atomspace(atomspace)
    scheme_eval(atomspace, "(use-modules (opencog))")
    scheme_eval(atomspace, "(use-modules (opencog exec))")
    scheme_eval(atomspace, "(use-modules (opencog query))")
    return atomspace

def runNeuralNetwork(boundingBox, conceptNode):
    log.debug('runNeuralNetwork: %s, %s', str(boundingBox), str(conceptNode))
    featuresValue = boundingBox.get_value(PredicateNode('features'))
    if featuresValue is None:
        log.debug('no features found, return FALSE')
        return TruthValue(0.0, 1.0)
    features = np.array(featuresValue.to_list())
    predicateName = conceptNode.name
    probability = GetFeatures.predicate(features, predicateName)
    log.debug('word: %s, result: %s', predicateName, str(probability))
    return TruthValue(probability, 1.0)

def answerQuestion(record):
    log.debug('processing question: %s', record.question)
    
    imageFileName = getImageFileName(record.imageId)
    imageFeatures = FloatValue(GetFeatures.getFeatures(imageFileName).tolist())

    atomspace.clear()
    boundingBoxInstance = ConceptNode('BoundingBox-1')
    InheritanceLink(boundingBoxInstance, ConceptNode('BoundingBox'))
    boundingBoxInstance.set_value(PredicateNode('features'), imageFeatures)
    
    relexFormula = questionConverter.parseQuestion(record.question)
    queryInScheme = questionConverter.convertToOpencogSchema(relexFormula)
    log.debug('Scheme query: %s', queryInScheme)

    evaluateStatement = '(cog-evaluate! ' + queryInScheme + ')'
    start = datetime.datetime.now()
    result = scheme_eval_v(atomspace, evaluateStatement)
    delta = datetime.datetime.now() - start
    print('The result of pattern matching is: ' + str(result) + 
          ', time: ' + str(delta.microseconds) + ' microseconds')


def initializeLogger():
    opencog.logger.log.set_level(opencogLogLevel)
    
    log = logging.getLogger('VqaMainLoop')
    log.setLevel(pythonLogLevel)
    log.addHandler(logging.StreamHandler())
    return log


def answerTestQuestion(question, imageId):
    questionRecord = Record()
    questionRecord.question = question
    questionRecord.imageId = imageId
    answerQuestion(questionRecord)


def answerAllQuestions(questionsFileName):
    questionFile = open(questionsFileName, 'r')
    for line in questionFile:
        try:
            record = Record.fromString(line)
            answerQuestion(record)
        except ValueError as ve:
            continue


def main():
    global log
    log = initializeLogger()
    
    log.info('VqaMainLoop started')
    
    question2atomeseLibraryPath = str(currentDir) + '/../question2atomese/target/question2atomese-1.0-SNAPSHOT.jar'
    jpype.startJVM(jpype.getDefaultJVMPath(), 
                   '-Djava.class.path=' + str(question2atomeseLibraryPath))
    global questionConverter
    questionConverter = jpype.JClass('org.opencog.vqa.relex.QuestionToOpencogConverter')()
    
    global atomspace
    atomspace = initializeAtomspace()
    
    answerTestQuestion('Is the hare grey?', '000001')
    
    jpype.shutdownJVM()
    
    log.info('VqaMainLoop stopped')


main()
