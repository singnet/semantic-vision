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
from opencog.scheme_wrapper import scheme_eval, scheme_eval_v

from netsvocabulary import NetsVocab

currentDir = os.path.dirname(os.path.realpath(__file__))
question2atomeeseLibraryPath = (str(currentDir) +
    '/../../question2atomeese/target/question2atomeese-1.0-SNAPSHOT.jar')

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
parser.add_argument('--words', '-w', dest='wordsFileName',
    action='store', type=str,
    help='words vocabulary (required only if old models file format is used')
parser.add_argument('--opencog-log-level', dest='opencogLogLevel',
    action='store', type = str, default='NONE',
    choices=['NONE', 'INFO', 'DEBUG', 'ERROR'],
    help='OpenCog logging level')
parser.add_argument('--python-log-level', dest='pythonLogLevel',
    action='store', type = str, default='INFO',
    choices=['INFO', 'DEBUG', 'ERROR'], 
    help='Python logging level')
parser.add_argument('--question2atomeese-java-library',
    dest='q2aJarFilenName', action='store', type = str,
    default=question2atomeeseLibraryPath,
    help='path to question2atomeese-<version>.jar')
args = parser.parse_args()

# global variables
opencogLogLevel = args.opencogLogLevel
pythonLogLevel = args.pythonLogLevel

log = None
questionConverter = None
atomspace = None
netsVocabulary = None

def addLeadingZeros(number, requriedLength):
    result = ''
    nZeros = int((requriedLength - 1) - math.floor(math.log10(number)))
    for _ in range(0, nZeros):
        result += '0'
    return result + str(number)

def getFeaturesFileName(imageId):
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
    atomspace = AtomSpace()
    set_type_ctor_atomspace(atomspace)
    scheme_eval(atomspace, "(use-modules (opencog))")
    scheme_eval(atomspace, "(use-modules (opencog exec))")
    scheme_eval(atomspace, "(use-modules (opencog query))")
    return atomspace

def runNeuralNetwork(boundingBox, conceptNode):
    log.debug('runNeuralNetwork: %s, %s', str(boundingBox), str(conceptNode))
    featuresValue = boundingBox.get_value(PredicateNode('features'))
    if (featuresValue is None):
        log.debug('no features found, return FALSE')
        return TruthValue(0.0, 1.0)
    features = np.array(featuresValue.to_list())
    word = conceptNode.name
    model = netsVocabulary.getModelByWord(word)
    # TODO: F.sigmoid should part of NN
    result = F.sigmoid(model(features))
    log.debug('word: %s, result: %s', word, str(result))
    return TruthValue(result.item(), 1.0)

def answerQuestion(record):
    log.debug('processing question: %s', record.question)
    atomspace.clear()
    
    featuresFileName = getFeaturesFileName(record.imageId)
    boundingBoxNumber = 0
    for boundingBoxFeatures in loadFeatures(featuresFileName):
        imageFeatures = FloatValue(boundingBoxFeatures)
        
        boundingBoxInstance = ConceptNode('BoundingBox-'+
                                          str(boundingBoxNumber))
        InheritanceLink(boundingBoxInstance, ConceptNode('BoundingBox'))
        boundingBoxInstance.set_value(PredicateNode('features'), imageFeatures)
        
        boundingBoxNumber += 1
    
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


# TODO reuse class from question2atomeese
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
    checkpoint = torch.load(args.modelsFileName, map_location=device.type)
    if ('version' in checkpoint['state_dict']):
        nets = NetsVocab(device)
        nets.load_state_dict(checkpoint['state_dict'])
    else:
        # Load vocabulary
        vocab = []
        with open(args.wordsFileName, 'r') as filehandle:
            vocab = [current_place.rstrip() for current_place in filehandle.readlines()]
        nets = NetsVocab(vocab, 2048, device)
        nets.load_state_dict_deprecated(checkpoint['state_dict'])
    return nets

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
    
#     answerAllQuestions(args.questionsFileName)
    answerTestQuestion('Are the zebras fat?', 11760)
    
    jpype.shutdownJVM()
    
    log.info('VqaMainLoop stopped')


main()
