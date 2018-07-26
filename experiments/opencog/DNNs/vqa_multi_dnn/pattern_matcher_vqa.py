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

### Reusable code (no dependency on global vars)

def addLeadingZeros(number, requriedLength):
    result = ''
    nZeros = int((requriedLength - 1) - math.floor(math.log10(int(number))))
    for _ in range(0, nZeros):
        result += '0'
    return result + str(number)

def loadDataFromZipOrFolder(folderOrZip, fileName, loadProcedure):
    if (os.path.isdir(folderOrZip)):
        with open(folderOrZip + '/' + fileName) as file:
            return loadProcedure(file)
    else:
        with zipfile.ZipFile(folderOrZip, 'r') as archive:
            with archive.open(fileName) as file:
                return loadProcedure(file)

def initializeLogger(opencogLogLevel, pythonLogLevel):
    opencog.logger.log.set_level(opencogLogLevel)
    
    log = logging.getLogger('VqaMainLoop')
    log.setLevel(pythonLogLevel)
    log.addHandler(logging.StreamHandler())
    return log

def initializeAtomspace(atomspaceFileName = None):
    atomspace = scheme_eval_as('(cog-atomspace)')
    scheme_eval(atomspace, '(use-modules (opencog))')
    scheme_eval(atomspace, '(use-modules (opencog exec))')
    scheme_eval(atomspace, '(use-modules (opencog query))')
    scheme_eval(atomspace, '(add-to-load-path ".")')
    if atomspaceFileName is not None:
        scheme_eval(atomspace, '(load-from-path "' + atomspaceFileName + '")')

    return atomspace

def pushAtomspace(parentAtomspace):
    scheme_eval(parentAtomspace, '(cog-push-atomspace)')
    childAtomspace = scheme_eval_as('(cog-atomspace)')
    set_type_ctor_atomspace(childAtomspace)
    return childAtomspace

def popAtomspace(childAtomspace):
    scheme_eval(childAtomspace, '(cog-pop-atomspace)')
    parentAtomspace = scheme_eval_as('(cog-atomspace)')
    set_type_ctor_atomspace(parentAtomspace)
    return parentAtomspace

def loadNets(modelsFileName):
    nets = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(modelsFileName, map_location=device.type)
    nets = NetsVocab.fromStateDict(device, checkpoint['state_dict'])
    return nets

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

class FeatureLoader:
    def loadFeaturesByImageId(self, imageId):
        pass

class TsvFileFeatureLoader(FeatureLoader):
    
    def __init__(self, featuresPath, featuresPrefix):
        self.featuresPath = featuresPath
        self.featuresPrefix = featuresPrefix
        
    def getFeaturesFileName(self, imageId):
        return self.featuresPrefix + addLeadingZeros(imageId, 12) + '.tsv'

    def loadFeaturesUsingFileHandle(self, fileHandle):
        featuresByBoundingBoxIndex = []
        next(fileHandle)
        for line in fileHandle:
            features = [float(number) for number in line.split()]
            featuresByBoundingBoxIndex.append(features[10:])
        return featuresByBoundingBoxIndex
    
    def loadFeaturesByFileName(self, featureFileName):
        return loadDataFromZipOrFolder(self.featuresPath, featureFileName, 
            lambda fileHandle: self.loadFeaturesUsingFileHandle(fileHandle))
        
    def loadFeaturesByImageId(self, imageId):
        return self.loadFeaturesByFileName(self.getFeaturesFileName(imageId))

class AnswerHandler:
    def onAnswer(self, record, answer):
        pass
    
class StatisticsAnswerHandler(AnswerHandler):
    
    def __init__(self):
        self.questionsAnswered = 0
        self.correctAnswers = 0

    def onAnswer(self, record, answer):
        self.questionsAnswered += 1
        if answer == record.answer:
            self.correctAnswers += 1
        log.debug('Correct answers %s%%', self.correctAnswerPercent())

    def correctAnswerPercent(self):
        return self.correctAnswers / self.questionsAnswered * 100

### Pipeline code

def runNeuralNetwork(boundingBox, conceptNode):
    log.debug('runNeuralNetwork: %s, %s', str(boundingBox), str(conceptNode))
    featuresValue = boundingBox.get_value(PredicateNode('features'))
    if featuresValue is None:
        log.debug('no features found, return FALSE')
        return TruthValue(0.0, 1.0)
    features = np.array(featuresValue.to_list())
    word = conceptNode.name
    global netsVocabulary
    model = netsVocabulary.getModelByWord(word)
    if model is None:
        log.debug('no model found, return FALSE')
        return TruthValue(0.0, 1.0)
    # TODO: F.sigmoid should part of NN
    result = F.sigmoid(model(torch.Tensor(features)))
    log.debug('word: %s, result: %s', word, str(result))
    # Return matching values from PatternMatcher by adding 
    # them to bounding box and concept node
    # TODO: how to return predicted values properly?
    boundingBox.set_value(conceptNode, FloatValue(result.item()))
    conceptNode.set_value(boundingBox, FloatValue(result.item()))
    return TruthValue(result.item(), 1.0)

class PatternMatcherVqaPipeline:
    
    def __init__(self, featureLoader, questionConverter, atomspace,
                 netsVocabulary, answerHandler):
        self.featureLoader = featureLoader
        self.questionConverter = questionConverter
        self.atomspace = atomspace
        self.netsVocabulary = netsVocabulary
        self.answerHandler = answerHandler

    # TODO: pass atomspace as parameter to exclude necessity of set_type_ctor_atomspace
    def addBoundingBoxesIntoAtomspace(self, record):
        boundingBoxNumber = 0
        for boundingBoxFeatures in self.featureLoader.loadFeaturesByImageId(record.imageId):
            imageFeatures = FloatValue(boundingBoxFeatures)
            boundingBoxInstance = ConceptNode(
                'BoundingBox-' + str(boundingBoxNumber))
            InheritanceLink(boundingBoxInstance, ConceptNode('BoundingBox'))
            boundingBoxInstance.set_value(PredicateNode('features'), imageFeatures)
            boundingBoxNumber += 1
    
    def answerQuestion(self, record):
        log.debug('processing question: %s', record.question)
        self.atomspace = pushAtomspace(self.atomspace)
        try:
            
            self.addBoundingBoxesIntoAtomspace(record)
            
            relexFormula = self.questionConverter.parseQuestion(record.question)
            queryInScheme = self.questionConverter.convertToOpencogScheme(relexFormula)
            if queryInScheme is None:
                log.debug('Question was not parsed')
                return
            log.debug('Scheme query: %s', queryInScheme)
        
            if record.questionType == 'yes/no':
                answer = self.answerYesNoQuestion(queryInScheme)
            else:
                answer = self.answerOtherQuestion(queryInScheme)
            
            self.answerHandler.onAnswer(record, answer)
            
            print('{}::{}::{}::{}::{}'.format(record.questionId, record.question, 
                answer, record.answer, record.imageId))
            
        finally:
            self.atomspace = popAtomspace(self.atomspace)
    
    def answerYesNoQuestion(self, queryInScheme):
        evaluateStatement = '(cog-evaluate! ' + queryInScheme + ')'
        start = datetime.datetime.now()
        result = scheme_eval_v(self.atomspace, evaluateStatement)
        delta = datetime.datetime.now() - start
        log.debug('The result of pattern matching is: %s, time: %s microseconds',
                  result, delta.microseconds)
        answer = 'yes' if result.to_list()[0] >= 0.5 else 'no'
        return answer
    
    class OtherQuestionResult:
        
        def __init__(self, bb, attribute, object):
            self.bb = bb
            self.attribute = attribute
            self.object = object
            self.attributeProbability = bb.get_value(attribute).to_list()[0]
            self.objectProbability = bb.get_value(object).to_list()[0]
            
        def __lt__(self, other):
            if abs(self.objectProbability - other.objectProbability) > 0.000001:
                return self.objectProbability < other.objectProbability
            else:
                return self.attributeProbability < other.attributeProbability
    
        def __gt__(self, other):
            return other.__lt__(self);
    
        def __str__(self):
            return '{} is {} {}({}), score = {}'.format(self.bb.name,
                                       self.attribute.name,
                                       self.object.name,
                                       self.objectProbability,
                                       self.attributeProbability)
    
    def answerOtherQuestion(self, queryInScheme):
        evaluateStatement = '(cog-execute! ' + queryInScheme + ')'
        start = datetime.datetime.now()
        resultsData = scheme_eval_h(self.atomspace, evaluateStatement)
        delta = datetime.datetime.now() - start
        log.debug('The resultsData of pattern matching is: %s, time: %s microseconds',
                  resultsData, delta.microseconds)
        
        results = []
        for resultData in resultsData.out:
            out = resultData.out
            results.append(self.OtherQuestionResult(out[0], out[1], out[2]))
        results.sort(reverse = True)
        
        for result in results:
            log.debug(str(result))
        
        maxResult = results[0]
        answer = maxResult.attribute.name
        return answer
    
    def answerTestQuestion(self, question, imageId):
        questionRecord = Record()
        questionRecord.question = question
        questionRecord.imageId = imageId
        self.answerQuestion(questionRecord)
    
    def answerAllQuestions(self, questionsFileName):
        questionFile = open(questionsFileName, 'r')
        for line in questionFile:
            try:
                record = Record.fromString(line)
                self.answerQuestion(record)
            except ValueError as ve:
                continue
    
### MAIN

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
parser.add_argument('--atomspace', '-a', dest='atomspaceFileName',
    action='store', type=str,
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
log = initializeLogger(args.opencogLogLevel, args.pythonLogLevel)
netsVocabulary = None

log.info('VqaMainLoop started')

jpype.startJVM(jpype.getDefaultJVMPath(), 
               '-Djava.class.path=' + str(args.q2aJarFilenName))
try:
    
    featureLoader = TsvFileFeatureLoader(args.featuresPath, args.featuresPrefix)
    questionConverter = jpype.JClass('org.opencog.vqa.relex.QuestionToOpencogConverter')()
    atomspace = initializeAtomspace(args.atomspaceFileName)
    netsVocabulary = loadNets(args.modelsFileName)
    statisticsAnswerHandler = StatisticsAnswerHandler()
    
    pmVqaPipeline = PatternMatcherVqaPipeline(featureLoader,
                                              questionConverter,
                                              atomspace,
                                              netsVocabulary,
                                              statisticsAnswerHandler)
    pmVqaPipeline.answerAllQuestions(args.questionsFileName)
    
    print('Questions answered: {}, correct answers: {}% ({})'
          .format(statisticsAnswerHandler.questionsAnswered,
                  statisticsAnswerHandler.correctAnswerPercent(),
                  statisticsAnswerHandler.correctAnswers))
finally:
    jpype.shutdownJVM()

log.info('VqaMainLoop stopped')
