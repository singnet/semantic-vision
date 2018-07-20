import os
import logging
import jpype
import numpy as np
import datetime
import opencog.logger
import argparse
import torch
import zipfile
import math
import torch.nn.functional as F

from opencog.atomspace import AtomSpace, TruthValue, types
from opencog.type_constructors import *
from opencog.scheme_wrapper import *

from netsvocabulary import NetsVocab

currentDir = os.path.dirname(os.path.realpath(__file__))
question2atomeseLibraryPath = (str(currentDir) +
    '/../../question2atomese/target/question2atomese-1.0-SNAPSHOT.jar')

parser = argparse.ArgumentParser(description='Load pretrained words models '
   'and answer questions using OpenCog PatternMatcher')
parser.add_argument('--questions', '-q', dest='questionsFileName',
    action='store', type=str, required=True,
    help='parsed questions file name')
parser.add_argument('--models', '-m', dest='modelsFileName',
    action='store', type=str, required=True,
    help='models file name')
parser.add_argument('--features', '-f', dest='featuresPath',
    action='store', type=str, required=True,
    help='features path (it can be either zip archive or folder name)')
parser.add_argument('--features-prefix', dest='featuresPrefix',
    action='store', type=str, default='val2014_parsed_features/COCO_val2014_',
    help='features prefix to be merged with path to open feature')
parser.add_argument('--facts', '-s', dest='factsFileName',
    action='store', type=str, required=True,
    help='Scheme program to fill atomspace with facts')
parser.add_argument('--opencog-log-level', dest='opencogLogLevel',
    action='store', type = str, default='NONE',
    choices=['FINE', 'DEBUG', 'INFO', 'ERROR', 'NONE'],
    help='OpenCog logging level')
parser.add_argument('--python-log-level', dest='pythonLogLevel',
    action='store', type = str, default='INFO',
    choices=['INFO', 'DEBUG', 'ERROR'], 
    help='Python logging level')
parser.add_argument('--question2atomese-java-library',
    dest='q2aJarFilenName', action='store', type = str,
    default=question2atomeseLibraryPath,
    help='path to question2atomese-<version>.jar')
args = parser.parse_args()

# global variables
opencogLogLevel = args.opencogLogLevel
pythonLogLevel = args.pythonLogLevel

log = None
questionConverter = None
atomspace = None
netsVocabulary = None
questionsAnswered = 0
correctAnswers = 0

def addLeadingZeros(number, requriedLength):
    result = ''
    nZeros = int((requriedLength - 1) - math.floor(math.log10(int(number))))
    for _ in range(0, nZeros):
        result += '0'
    return result + str(number)

def getFeaturesFileName(imageId):
    global args
    return args.featuresPrefix + addLeadingZeros(imageId, 12) + '.tsv'

def loadDataFromZipOrFolder(folderOrZip, fileName, loadProcedure):
    if (os.path.isdir(folderOrZip)):
        with open(folderOrZip + '/' + fileName) as file:
            return loadProcedure(file)
    else:
        with zipfile.ZipFile(folderOrZip, 'r') as archive:
            with archive.open(fileName) as file:
                return loadProcedure(file)

def loadFeaturesUsingFileHandle(fileHandle):
    featuresByBoundingBoxIndex = []
    next(fileHandle)
    for line in fileHandle:
        features = [float(number) for number in line.split()]
        featuresByBoundingBoxIndex.append(features[10:])
    return featuresByBoundingBoxIndex

def loadFeatures(featureFileName):
    return loadDataFromZipOrFolder(args.featuresPath, featureFileName, 
                            loadFeaturesUsingFileHandle)

def initializeAtomspace():
    atomspace = scheme_eval_as('(cog-atomspace)')
    scheme_eval(atomspace, '(use-modules (opencog))')
    scheme_eval(atomspace, '(use-modules (opencog exec))')
    scheme_eval(atomspace, '(use-modules (opencog query))')
    scheme_eval(atomspace, '(add-to-load-path ".")')
    global args
    if args.factsFileName is not None:
        scheme_eval(atomspace, '(load-from-path "' + args.factsFileName + '")')

    return atomspace

def runNeuralNetwork(boundingBox, conceptNode):
    log.debug('runNeuralNetwork: %s, %s', str(boundingBox), str(conceptNode))
    featuresValue = boundingBox.get_value(PredicateNode('features'))
    if featuresValue is None:
        log.debug('no features found, return FALSE')
        return TruthValue(0.0, 1.0)
    features = np.array(featuresValue.to_list())
    word = conceptNode.name
    model = netsVocabulary.getModelByWord(word)
    if model is None:
        log.debug('no model found, return FALSE')
        return TruthValue(0.0, 1.0)
    # TODO: F.sigmoid should part of NN
    result = F.sigmoid(model(torch.Tensor(features)))
    log.debug('word: %s, result: %s', word, str(result))
    return TruthValue(result.item(), 1.0)

def pushAtomspace():
    global atomspace
    scheme_eval(atomspace, '(cog-push-atomspace)')
    atomspace = scheme_eval_as('(cog-atomspace)')
    set_type_ctor_atomspace(atomspace)

def popAtomspace():
    global atomspace
    scheme_eval(atomspace, '(cog-pop-atomspace)')
    atomspace = scheme_eval_as('(cog-atomspace)')
    set_type_ctor_atomspace(atomspace)

# TODO: pass atomspace as parameter to exclude necessity of set_type_ctor_atomspace
def addBoundingBoxesIntoAtomspace(record):
    featuresFileName = getFeaturesFileName(record.imageId)
    boundingBoxNumber = 0
    for boundingBoxFeatures in loadFeatures(featuresFileName):
        imageFeatures = FloatValue(boundingBoxFeatures)
        boundingBoxInstance = ConceptNode(
            'BoundingBox-' + str(boundingBoxNumber))
        InheritanceLink(boundingBoxInstance, ConceptNode('BoundingBox'))
        boundingBoxInstance.set_value(PredicateNode('features'), imageFeatures)
        boundingBoxNumber += 1

def answerQuestion(record):
    log.debug('processing question: %s', record.question)
    pushAtomspace()
    try:
        
        addBoundingBoxesIntoAtomspace(record)
        
        relexFormula = questionConverter.parseQuestion(record.question)
        queryInScheme = questionConverter.convertToOpencogScheme(relexFormula)
        if queryInScheme is None:
            log.debug('Question was not parsed')
            return
        log.debug('Scheme query: %s', queryInScheme)
    
        if record.questionType == 'yes/no':
            answer = answerYesNoQuestion(queryInScheme)
        else:
            answer = answerOtherQuestion(queryInScheme)
        
        global questionsAnswered, correctAnswers
        questionsAnswered += 1
        if answer == record.answer:
            correctAnswers += 1
        
        log.debug('Correct answers %s%%', correctAnswerPercent())
        print('{}::{}::{}::{}::{}'.format(record.questionId, record.question, 
            answer, record.answer, record.imageId))
        
    finally:
        popAtomspace()

def answerYesNoQuestion(queryInScheme):
    evaluateStatement = '(cog-evaluate! ' + queryInScheme + ')'
    start = datetime.datetime.now()
    global atomspace
    result = scheme_eval_v(atomspace, evaluateStatement)
    delta = datetime.datetime.now() - start
    log.debug('The result of pattern matching is: %s, time: %s microseconds',
              result, delta.microseconds)
    answer = 'yes' if result.to_list()[0] >= 0.5 else 'no'
    return anwer

def answerOtherQuestion(queryInScheme):
    evaluateStatement = '(cog-execute! ' + queryInScheme + ')'
    start = datetime.datetime.now()
    global atomspace
    result = scheme_eval_v(atomspace, evaluateStatement)
    delta = datetime.datetime.now() - start
    log.debug('The result of pattern matching is: %s, time: %s microseconds',
              result, delta.microseconds)
    answer = None # TODO: get answer from matching results
    return answer

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


# TODO reuse class from question2atomese
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
         record.formula, record.groundedFormula) = string.strip().split('::')
        return record
    
    def getWords(self):
        # parse '_test(A, B);next(B, A)'
        words = re.split('\)[^\(]+\(|, |^[^\(]+\(|\)[^\(]+$', 
                         self.groundedFormula)
        return map(str.strip, words)

def answerAllQuestions(questionsFileName):
    questionFile = open(questionsFileName, 'r')
    for line in questionFile:
        try:
            record = Record.fromString(line)
            answerQuestion(record)
        except ValueError as ve:
            continue


def loadNets():
    nets = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global args
    checkpoint = torch.load(args.modelsFileName, map_location=device.type)
    nets = NetsVocab.fromStateDict(device, checkpoint['state_dict'])
    return nets


def correctAnswerPercent():
    return correctAnswers / questionsAnswered * 100

def main():
    global log
    log = initializeLogger()
    
    log.info('VqaMainLoop started')
    
    jpype.startJVM(jpype.getDefaultJVMPath(), 
                   '-Djava.class.path=' + str(args.q2aJarFilenName))
    global questionConverter
    questionConverter = jpype.JClass('org.opencog.vqa.relex.QuestionToOpencogConverter')()
    
    global atomspace
    atomspace = initializeAtomspace()
    
    global netsVocabulary
    netsVocabulary = loadNets()
    
    answerAllQuestions(args.questionsFileName)
#     answerTestQuestion('What color is the sky?', 11760)
    print('Questions answered: {}, correct answers: {}% ({})'
          .format(questionsAnswered,
                  correctAnswerPercent(),
                  correctAnswers))
    
    jpype.shutdownJVM()
    
    log.info('VqaMainLoop stopped')


main()
