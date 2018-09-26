import sys
import logging
import datetime
import argparse

import jpype
import numpy as np
import opencog.logger
import network_runner

from opencog.atomspace import TruthValue
from opencog.type_constructors import *
from opencog.scheme_wrapper import *

from util import *
from interface import FeatureExtractor, AnswerHandler, NoModelException
from multidnn import NetsVocabularyNeuralNetworkRunner
from hypernet import HyperNetNeuralNetworkRunner
from splitnet.splitmultidnnmodel import SplitMultidnnRunner


sys.path.insert(0, currentDir(__file__) + '/../question2atomese')
from record import Record

logger = logging.getLogger(__name__)
### Reusable code (no dependency on global vars)

def initializeRootAndOpencogLogger(opencogLogLevel, pythonLogLevel):
    opencog.logger.log.set_level(opencogLogLevel)

    rootLogger = logging.getLogger()
    rootLogger.setLevel(pythonLogLevel)
    rootLogger.addHandler(logging.StreamHandler())


def pushAtomspace(parentAtomspace):
    # TODO: cannot push/pop atomspace via Python API,
    # workarouding it using Scheme API
    scheme_eval(parentAtomspace, '(cog-push-atomspace)')
    childAtomspace = scheme_eval_as('(cog-atomspace)')
    set_type_ctor_atomspace(childAtomspace)
    return childAtomspace


def popAtomspace(childAtomspace):
    scheme_eval(childAtomspace, '(cog-pop-atomspace)')
    parentAtomspace = scheme_eval_as('(cog-atomspace)')
    set_type_ctor_atomspace(parentAtomspace)
    return parentAtomspace


class TsvFileFeatureLoader(FeatureExtractor):

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

    def getFeaturesByImageId(self, imageId):
        return self.loadFeaturesByFileName(self.getFeaturesFileName(imageId))


class StatisticsAnswerHandler(AnswerHandler):

    def __init__(self):
        self.logger = logging.getLogger('StatisticsAnswerHandler')
        self.processedQuestions = 0
        self.questionsAnswered = 0
        self.correctAnswers = 0
        self.dont_know = 0
        self._dont_know_queries = []

    def onNewQuestion(self, record):
        self.processedQuestions += 1

    def onAnswer(self, record, answer):
        self.questionsAnswered += 1
        if answer == record.answer:
            self.correctAnswers += 1
        if answer is None:
            self.dont_know += 1
            self._dont_know_queries.append(record)
        self.logger.debug('Correct answers %s%%', self.correctAnswerPercent())

    def correctAnswerPercent(self):
        return self.correctAnswers / self.questionsAnswered * 100

    def unanswered_percent(self):
        return self.dont_know / self.questionsAnswered * 100

    def getUnanswered(self):
        return self._dont_know_queries


### Pipeline code

def runNeuralNetwork(boundingBox, conceptNode):
    logger = logging.getLogger('runNeuralNetwork')
    try:
        logger.debug('runNeuralNetwork: %s, %s', boundingBox.name, conceptNode.name)

        featuresValue = boundingBox.get_value(PredicateNode('features'))
        if featuresValue is None:
            logger.error('no features found, return FALSE')
            return TruthValue(0.0, 0.0)
        features = np.array(featuresValue.to_list())
        word = conceptNode.name

        certainty = 1.0
        neuralNetworkRunner = network_runner.runner
        try:
            resultTensor = neuralNetworkRunner.runNeuralNetwork(features, word)
        except NoModelException as e:
            import torch
            resultTensor = torch.zeros(1)
            certainty = 0.0
        result = resultTensor.item()

        logger.debug('bb: %s, word: %s, result: %s', boundingBox.name, word, str(result))
        # Return matching values from PatternMatcher by adding
        # them to bounding box and concept node
        # TODO: how to return predicted values properly?
        boundingBox.set_value(conceptNode, FloatValue(result))
        conceptNode.set_value(boundingBox, FloatValue(result))
        return TruthValue(result, certainty)
    except BaseException as e:
        logger.exception('Unexpected exception %s', e)
        return TruthValue(0.0, 1.0)


class OtherDetSubjObjResult:

    def __init__(self, bounding_box, attribute, object):
        self.bb = bounding_box
        self.attribute = attribute
        self.object = object
        self.attributeProbability = bounding_box.get_value(attribute).to_list()[0]
        self.objectProbability = bounding_box.get_value(object).to_list()[0]

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


class PatternMatcherVqaPipeline:

    def __init__(self, featureExtractor, questionConverter, atomspace, answerHandler):
        self.featureExtractor = featureExtractor
        self.questionConverter = questionConverter
        self.atomspace = atomspace
        self.answerHandler = answerHandler
        self.formulaQuestionMap = dict()
        self.logger = logging.getLogger('PatternMatcherVqaPipeline')

    # TODO: pass atomspace as parameter to exclude necessity of set_type_ctor_atomspace
    def addBoundingBoxesIntoAtomspace(self, record=None, image=None):
        """
        populate atomspace with bounding boxes, that is concept nodes

        Each bounding box node holds FloatValue, which in turn holds array of float values -
        activations of neural network on the corresponding image area.

        Parameters
        ----------
        record : Record
            record object, holding imageId. Features will be obtained using this id
        image : numpy.array
            an image to process. Features will be obtained directly from image

        Returns
        -------
        None
        """

        assert (record is None) != (image is None)
        if image is None:
            features = self.featureExtractor.getFeaturesByImageId(record.imageId)
        else:
            features = self.featureExtractor.getFeaturesByImage(image)
        boundingBoxNumber = 0
        for boundingBoxFeatures in features:
            imageFeatures = FloatValue(boundingBoxFeatures)
            boundingBoxInstance = ConceptNode(
                'BoundingBox-' + str(boundingBoxNumber))
            InheritanceLink(boundingBoxInstance, ConceptNode('BoundingBox'))
            boundingBoxInstance.set_value(PredicateNode('features'), imageFeatures)
            boundingBoxNumber += 1

    def get_question_type(self, formula):
        for formula_key, question_type in self.formulaQuestionMap.items():
            if formula_key in formula:
                return question_type
        raise NotImplementedError("formula({0}) is not supported yet".format(formula))

    def answerQuery(self, questionType, query):
        if questionType == 'yes/no':
            answer = self.answerYesNoQuestion(query)
        else:
            answer = self.answerOtherQuestion(query)
        return answer

    def answerQuestionByImage(self, image, question):
        self.atomspace = pushAtomspace(self.atomspace)
        try:
            self.addBoundingBoxesIntoAtomspace(image=image)
            parsedQuestion = self.questionConverter.parseQuestionAndType(question)
            relexFormula = parsedQuestion.relexFormula
            queryInScheme = self.questionConverter.convertToOpencogScheme(relexFormula)
            if queryInScheme is None:
                self.logger.error('Question was not parsed')
                return
            self.logger.debug('Scheme query: %s', queryInScheme)
            questionType = parsedQuestion.questionType
            return self.answerQuery(questionType, queryInScheme)
        finally:
            self.atomspace = popAtomspace(self.atomspace)

    def answerQuestion(self, record):
        self.logger.debug('processing question: %s', record.question)
        self.answerHandler.onNewQuestion(record)
        # Push/pop atomspace each time to not pollute it by temporary
        # bounding boxes
        self.atomspace = pushAtomspace(self.atomspace)
        try:
            self.addBoundingBoxesIntoAtomspace(record=record)

            relexFormula = self.questionConverter.parseQuestion(record.question)
            queryInScheme = self.questionConverter.convertToOpencogScheme(relexFormula)
            if queryInScheme is None:
                self.logger.error('Question was not parsed')
                return
            self.logger.debug('Scheme query: %s', queryInScheme)
            answer = self.answerQuery(record.questionType, queryInScheme)
            self.answerHandler.onAnswer(record, answer)

            print('{}::{}::{}::{}::{}'.format(record.questionId, record.question,
                answer, record.answer, record.imageId))

        finally:
            self.atomspace = popAtomspace(self.atomspace)

    def answerYesNoQuestion(self, queryInScheme):
        evaluateStatement = '(cog-evaluate! ' + queryInScheme + ')'
        start = datetime.datetime.now()
        # TODO: evaluates AND operations as crisp logic values
        # if clause has tv->mean() > 0.5 then return tv->mean() == 1.0
        result = scheme_eval_v(self.atomspace, evaluateStatement)
        delta = datetime.datetime.now() - start
        self.logger.debug('The result of pattern matching is: '
                          '%s, time: %s microseconds',
                          result, delta.microseconds)
        # TODO: Python value API improvements: convert single value to
        # appropriate type directly?
        answer = 'yes' if result.to_list()[0] >= 0.5 else 'no'
        return answer

    def answerOtherQuestion(self, queryInScheme):
        """
        Find answer for question with formula _det(A, B);_obj(C, D);_subj(C, A)

        :param queryInScheme: str
            query to pattern matcher or ure
        :return: str or None
            str if answer was found None otherwise
        """
        evaluateStatement = '(cog-execute! ' + queryInScheme + ')'
        start = datetime.datetime.now()
        resultsData = scheme_eval_h(self.atomspace, evaluateStatement)
        delta = datetime.datetime.now() - start
        self.logger.debug('The resultsData of pattern matching contains: '
                          '%s records, time: %s microseconds',
                          len(resultsData.out), delta.microseconds)

        results = []
        for resultData in resultsData.out:
            out = resultData.out
            results.append(OtherDetSubjObjResult(out[0], out[1], out[2]))
        results.sort(reverse=True)

        for result in results:
            self.logger.debug(str(result))
        if not results:
            return None
        maxResult = results[0]
        answer = maxResult.attribute.name
        return answer

    def answerSingleQuestion(self, question, imageId):
        questionRecord = Record()
        questionRecord.question = question
        questionRecord.imageId = imageId
        self.answerQuestion(questionRecord)

    @classmethod
    def is_record(cls, line):
        striped_line = line.strip()
        if striped_line.startswith('#'):
            return False
        return True

    def answerQuestionsFromFile(self, questionsFileName):
        questionFile = open(questionsFileName, 'r')
        for line in questionFile:
            if not self.is_record(line):
                continue
            try:
                record = Record.fromString(line)
                self.answerQuestion(record)
            except BaseException as e:
                logger.exception('Unexpected exception %s', e)
                continue

### MAIN
question2atomeseLibraryPath = (currentDir(__file__) +
    '/../question2atomese/target/question2atomese-1.0-SNAPSHOT.jar')


def parse_args():
    parser = argparse.ArgumentParser(description='Load pretrained words models '
       'and answer questions using OpenCog PatternMatcher')
    parser.add_argument('--model-kind', '-k', dest='kindOfModel',
        action='store', type=str, required=True,
        choices=['MULTIDNN', 'HYPERNET', 'SPLITMULTIDNN'],
        help='model kind: (1) MULTIDNN and SPLITMULTIDNN requires --model parameter only; '
        '(2) HYPERNET requires --model, --words and --embedding parameters')
    parser.add_argument('--questions', '-q', dest='questionsFileName',
        action='store', type=str, required=True,
        help='parsed questions file name')
    parser.add_argument('--multidnn-model', dest='multidnnModelFileName',
        action='store', type=str,
        help='Multi DNN model file name')
    parser.add_argument('--hypernet-model', dest='hypernetModelFileName',
        action='store', type=str,
        help='Hypernet model file name')
    parser.add_argument('--hypernet-words', '-w', dest='hypernetWordsFileName',
        action='store', type=str,
        help='words dictionary')
    parser.add_argument('--hypernet-embeddings', '-e',dest='hypernetWordEmbeddingsFileName',
        action='store', type=str,
        help='word embeddings')
    parser.add_argument('--features-extractor-kind', dest='kindOfFeaturesExtractor',
        action='store', type=str, required=True,
        choices=['PRECALCULATED', 'IMAGE'],
        help='features extractor type: (1) PRECALCULATED loads precalculated features; '
        '(2) IMAGE extract features from images on the fly')
    parser.add_argument('--precalculated-features', '-f', dest='precalculatedFeaturesPath',
        action='store', type=str,
        help='precalculated features path (it can be either zip archive or folder name)')
    parser.add_argument('--precalculated-features-prefix', dest='precalculatedFeaturesPrefix',
        action='store', type=str, default='val2014_parsed_features/COCO_val2014_',
        help='precalculated features prefix to be merged with path to open feature')
    parser.add_argument('--images', '-i', dest='imagesPath',
        action='store', type=str,
        help='path to images, required only when featur')
    parser.add_argument('--images-prefix', dest='imagesPrefix',
        action='store', type=str, default='val2014/COCO_val2014_',
        help='image file prefix to be merged with path to open image')
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
    return args


def main():
    args = parse_args()
    initializeRootAndOpencogLogger(args.opencogLogLevel, args.pythonLogLevel)

    logger = logging.getLogger('PatternMatcherVqaTest')
    logger.info('VqaMainLoop started')

    jpype.startJVM(jpype.getDefaultJVMPath(),
                   '-Djava.class.path=' + str(args.q2aJarFilenName))
    try:

        if args.kindOfFeaturesExtractor == 'IMAGE':
            from feature.image import ImageFeatureExtractor
            featureExtractor = ImageFeatureExtractor(
                # TODO: replace by arguments
                '/mnt/fileserver/shared/vital/image-features/test.prototxt',
                '/mnt/fileserver/shared/vital/image-features/resnet101_faster_rcnn_final_iter_320000_for_36_bboxes.caffemodel',
                args.imagesPath,
                args.imagesPrefix
                )
        elif args.kindOfFeaturesExtractor == 'PRECALCULATED':
            featureExtractor = TsvFileFeatureLoader(args.precalculatedFeaturesPath,
                                             args.precalculatedFeaturesPrefix)
        else:
            raise ValueError('Unexpected args.kindOfFeaturesExtractor value: {}'
                             .format(args.kindOfFeaturesExtractor))

        questionConverter = jpype.JClass('org.opencog.vqa.relex.QuestionToOpencogConverter')()
        atomspace = initialize_atomspace(args.atomspaceFileName)
        statisticsAnswerHandler = StatisticsAnswerHandler()
        if (args.kindOfModel == 'MULTIDNN'):
            network_runner.runner = NetsVocabularyNeuralNetworkRunner(args.multidnnModelFileName)
        elif (args.kindOfModel == 'SPLITMULTIDNN'):
            network_runner.runner = SplitMultidnnRunner(args.multidnnModelFileName)
        elif (args.kindOfModel == 'HYPERNET'):
            network_runner.runner = HyperNetNeuralNetworkRunner(args.hypernetWordsFileName,
                            args.hypernetWordEmbeddingsFileName, args.hypernetModelFileName)
        else:
            raise ValueError('Unexpected args.kindOfModel value: {}'.format(args.kindOfModel))

        pmVqaPipeline = PatternMatcherVqaPipeline(featureExtractor,
                                                  questionConverter,
                                                  atomspace,
                                                  statisticsAnswerHandler)
        pmVqaPipeline.answerQuestionsFromFile(args.questionsFileName)

        print('Questions processed: {0}, answered: {1}, correct answers: {2}% ({3}), unaswered {4}%'
              .format(statisticsAnswerHandler.processedQuestions,
                      statisticsAnswerHandler.questionsAnswered,
                      statisticsAnswerHandler.correctAnswerPercent(),
                      statisticsAnswerHandler.correctAnswers,
                      statisticsAnswerHandler.unanswered_percent()))
        with open("unanswered.txt", 'w') as f:
            for record in statisticsAnswerHandler.getUnanswered():
                f.write(record.toString() + '\n')
    finally:
        jpype.shutdownJVM()

    logger.info('VqaMainLoop stopped')

if __name__ == '__main__':
    main()
