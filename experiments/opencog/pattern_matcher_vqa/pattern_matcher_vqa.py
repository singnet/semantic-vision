import sys
import logging
import datetime
import argparse
import abc

import jpype
import numpy as np
import opencog.logger
import network_runner

from opencog.atomspace import TruthValue
from opencog.type_constructors import *
from opencog.scheme_wrapper import *
from opencog import bindlink

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
    """Create child atomspace"""
    # TODO: cannot push/pop atomspace via Python API,
    # workarouding it using Scheme API
    scheme_eval(parentAtomspace, '(cog-push-atomspace)')
    childAtomspace = scheme_eval_as('(cog-atomspace)')
    set_type_ctor_atomspace(childAtomspace)
    return childAtomspace


def popAtomspace(childAtomspace):
    """Destroy child atomspace"""
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
    """
    Callback for running from within the atomspace from ground predicate

    :param boundingBox: Atom
    :param conceptNode: Atom
    :return: TruthValue
    """
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
        groundedPredicate = GroundedPredicateNode("py:runNeuralNetwork")
        ev = EvaluationLink(groundedPredicate, ListLink(boundingBox, conceptNode))
        tv = TruthValue(result, certainty)
        ev.tv = tv
        return tv
    except BaseException as e:
        logger.exception('Unexpected exception %s', e)
        return TruthValue(0.0, 1.0)


class OtherDetSubjObj:
    @abc.abstractmethod
    def get_predicate_name(self):
        pass

    def get_bounding_box_id(self):
        pass

    def get_expression(self):
        pass


class QueryProcessingData:
    """
    Holds different intermediate results for query computation
    """
    __slots__ = ["relexFormula", "query", "answer", "boundingBoxes", "answerBox", "answerExpression"]

    def __init__(self, relexFormula, query, answer, boundingBoxes, answerBox, answerExpression):
        self.relexFormula = relexFormula
        self.query = query
        self.answer = answer
        self.boundingBoxes = boundingBoxes
        self.answerBox = answerBox
        self.answerExpression = answerExpression

    def __str__(self):
        return self.answer

    def __repr__(self):
        return "QueryProcessingData({0}, \n{1}, \n{2}, \n{3},\n {4},\n {5})".format(self.relexFormula,
                                                                                    self.query,
                                                                                    self.answer,
                                                                                    self.boundingBoxes,
                                                                                    self.answerBox,
                                                                                    self.answerExpression)


class OtherDetSubjObjResult(OtherDetSubjObj):

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

    def get_predicate_name(self):
        return self.attribute.name


class ConjunctionResult(OtherDetSubjObj):
    def __init__(self, atom):
        self.atom = atom
        self.strength = atom.tv.mean
        self.confidence = atom.tv.confidence

    def __lt__(self, other):
        if abs(self.strength - other.strength) > 0.000001:
            return self.strength < other.strength
        else:
            return self.confidence < other.confidence

    def __gt__(self, other):
        return other.__lt__(self)

    def get_predicate_name(self):
        return extract_predicate(self.atom.out)

    def get_bounding_box_id(self):
        return extract_bb_id(self.atom.out)

    def get_expression(self):
        return self.atom


def extract_predicate(atoms):
    for atom in atoms:
        if atom.type == opencog.atomspace.types.InheritanceLink:
            if atom.out[1].name != 'BoundingBox':
                predicate_name = atom.out[0].name
                return predicate_name


def extract_bb_id(atoms):
    for atom in atoms:
        if atom.type == opencog.atomspace.types.InheritanceLink:
            if atom.out[1].name == 'BoundingBox':
                predicate_name = atom.out[0].name
                return int(predicate_name.split('-')[-1])


class PatternMatcherVqaPipeline:

    def __init__(self, featureExtractor, questionConverter, atomspace, answerHandler):
        """
        Construct pattern matcher object

        :param featureExtractor: interface.FeatureExtractor
            feature extractor for images
        :param questionConverter: org.opencog.vqa.relex.QuestionToOpencogConverter
            converter from text to atomese queries
        :param atomspace: atomspace
            atomspace with background knowledge
        :param answerHandler: interface.AnswerHandler
            answer handler for statistics
        """
        self.featureExtractor = featureExtractor
        self.questionConverter = questionConverter
        self.atomspace = atomspace
        self.answerHandler = answerHandler
        self.logger = logging.getLogger('PatternMatcherVqaPipeline')

    # TODO: pass atomspace as parameter to exclude necessity of set_type_ctor_atomspace
    def addBoundingBoxesIntoAtomspace(self, features):
        """
        populate atomspace with bounding boxes, that is concept nodes

        Each bounding box node holds FloatValue, which in turn holds array of float values -
        activations of neural network on the corresponding image area.

        Parameters
        ----------
        features : Iterable
            iterable with bounding box features

        Returns
        -------
        None
        """
        boundingBoxNumber = 0
        for boundingBoxFeatures in features:
            imageFeatures = FloatValue(boundingBoxFeatures)
            boundingBoxInstance = ConceptNode(
                'BoundingBox-' + str(boundingBoxNumber))
            inh = InheritanceLink(boundingBoxInstance, ConceptNode('BoundingBox'))
            tv = TruthValue(1.0, 1.0)
            inh.tv = tv
            boundingBoxInstance.set_value(PredicateNode('features'), imageFeatures)
            boundingBoxNumber += 1

    def answerQuery(self, questionType, query):
        if questionType == 'yes/no':
            answer = self.answerYesNoQuestion(query)
        else:
            answer = self.answerOtherQuestion(query)
        return answer

    def answerQuestionByImage(self, image, question, use_pm=True) -> QueryProcessingData:
        """
        Get answer from image and text
        :param image: numpy.array
            height x width x num channels
            expected type = numpy.uint8
        :param question: str
        :param use_pm: bool
            if use_pm == True, pattern matcher will be used to compute the answer
            otherwise unified rule engine will be used
        :return: QueryProcessingData
        """
        self.atomspace = pushAtomspace(self.atomspace)
        try:
            features, boxes = self.featureExtractor.getFeaturesByImage(image)
            self.addBoundingBoxesIntoAtomspace(features)
            parsedQuestion = self.questionConverter.parseQuestionAndType(question)
            relexFormula = parsedQuestion.relexFormula
            if use_pm:
                queryInScheme = self.questionConverter.convertToOpencogSchemePM(relexFormula)
            else:
                queryInScheme = self.questionConverter.convertToOpencogSchemeURE(relexFormula)
            if queryInScheme is None:
                self.logger.error('Question was not parsed')
                return
            self.logger.debug('Scheme query: %s', queryInScheme)
            questionType = parsedQuestion.questionType
            if questionType is None:
                return
            answer, bb_id, expr = self.answerQuery(questionType, queryInScheme)
            result = QueryProcessingData(relexFormula, queryInScheme, answer, boxes,
                                         answerBox=bb_id, answerExpression=expr)
            return result
        finally:
            self.atomspace = popAtomspace(self.atomspace)

    def answerQuestion(self, record, use_pm=True):
        self.logger.debug('processing question: %s', record.question)
        self.answerHandler.onNewQuestion(record)
        # Push/pop atomspace each time to not pollute it by temporary
        # bounding boxes
        self.atomspace = pushAtomspace(self.atomspace)
        try:
            features = self.featureExtractor.getFeaturesByImageId(record.imageId)
            self.addBoundingBoxesIntoAtomspace(features)

            relexFormula = self.questionConverter.parseQuestion(record.question)
            if use_pm:
                queryInScheme = self.questionConverter.convertToOpencogSchemePM(relexFormula)
            else:
                queryInScheme = self.questionConverter.convertToOpencogSchemeURE(relexFormula)
            if queryInScheme is None:
                self.logger.error('Question was not parsed')
                return
            self.logger.debug('Scheme query: %s', queryInScheme)
            answer, _, _ = self.answerQuery(record.questionType, queryInScheme)
            self.answerHandler.onAnswer(record, answer)

            print('{}::{}::{}::{}::{}'.format(record.questionId, record.question,
                answer, record.answer, record.imageId))

        finally:
            self.atomspace = popAtomspace(self.atomspace)

    def answerYesNoQuestion(self, queryInScheme):
        """
        Find answer for question with formula _predadj(A, B)

        :param queryInScheme: str
            query to pattern matcher or ure
        :return: Tuple[str, int, str]
            if answer was found Tuple[None, None, None] otherwise
        """
        start = datetime.datetime.now()

        result = scheme_eval_h(self.atomspace, queryInScheme)
        delta = datetime.datetime.now() - start
        self.logger.debug('The result of pattern matching is: '
                          '%s, time: %s microseconds',
                          result, delta.microseconds)
        results = self.sort_results(result, a_extract_predicate=False)
        if not results:
            return 'no', None, None
        maxResult = results[0]
        bb_id = maxResult.get_bounding_box_id()
        expression = maxResult.get_expression()
        answer = 'yes' if expression.tv.mean > 0.5 else 'no'
        return answer, bb_id, expression

    def answerOtherQuestion(self, queryInScheme):
        """
        Find answer for question with formula _det(A, B);_obj(C, D);_subj(C, A)

        :param queryInScheme: str
            query to pattern matcher or ure
        :return: Tuple[str, int, str]
            if answer was found Tuple[None, None, None] otherwise
        """
        start = datetime.datetime.now()
        resultsData = scheme_eval_h(self.atomspace, queryInScheme)
        delta = datetime.datetime.now() - start
        self.logger.debug('The resultsData of pattern matching contains: '
                          '%s records, time: %s microseconds',
                          len(resultsData.out), delta.microseconds)

        results = self.sort_results(resultsData, a_extract_predicate=True)
        if not results:
            return None, None, None
        maxResult = results[0]
        return maxResult.get_predicate_name(), \
               maxResult.get_bounding_box_id(), \
               maxResult.get_expression()

    def sort_results(self, resultsData, a_extract_predicate=False):
        results = []
        for resultData in resultsData.out:
            out = resultData.out
            if resultData.type == opencog.atomspace.types.AndLink:
                # resultData is AndLink with random order of conjucts
                results.append(ConjunctionResult(resultData))
            else:
                results.append(OtherDetSubjObjResult(out[0], out[1], out[2]))
        results.sort(reverse=True)
        for result in results:
            self.logger.debug(str(result))
        return results

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

    def answerQuestionsFromFile(self, questionsFileName, use_pm=True):
        questionFile = open(questionsFileName, 'r')
        for line in questionFile:
            if not self.is_record(line):
                continue
            try:
                record = Record.fromString(line)
                self.answerQuestion(record, use_pm=use_pm)
            except BaseException as e:
                logger.exception('Unexpected exception %s', e)
                continue


class TBD(PatternMatcherVqaPipeline):


    def answerByPrograms(self, tdb_net, features, programs):
        batch_size = features.size(0)
        feat_input_volume = tdb_net.stem(features)
        results = []
        for n in range(batch_size):
            feat_input = feat_input_volume[n:n + 1]
            output = feat_input
            program = []
            for i in reversed(programs.data[n].cpu().numpy()):
                module_type = tdb_net.vocab['program_idx_to_token'][i]
                if module_type == '<NULL>':
                    continue
                program.append(module_type)
            result = self.run_program(output, program)
            results.append(result)
        return results

    def argmax(self, answer_set):
        import tbd_helpers
        items = []
        key_attention, key_scene, key_shape_attention, key_shape_scene = tbd_helpers.generate_keys(self.atomspace)
        for list_link in answer_set.get_out():
            atoms = list_link.get_out()
            value = -1
            concept = None
            for atom in atoms:
                if atom.name.startswith("Data-"):
                    value = tbd_helpers.extract_tensor(atom, key_attention, key_shape_attention).numpy().sum()
                elif atom.name.startswith("BoundingBox"):
                    continue
                else:
                    concept = atom.name
            assert concept
            assert value != -1
            items.append((value, concept))
        if not items:
            return None
        items.sort(reverse=True)
        return items[0][1]

    def run_program(self, features, program):
        self.atomspace = pushAtomspace(self.atomspace)
        self._add_scene_atom(features)
        import tbd_helpers
        eval_link, left, inheritance_set = tbd_helpers.return_prog(commands=tuple(reversed(program)), atomspace=self.atomspace)
        bind_link = tbd_helpers.build_bind_link(self.atomspace, eval_link, inheritance_set)
        result = bindlink.bindlink(self.atomspace, bind_link)
        return self.argmax(result)

    def _add_scene_atom(self, features):
        import tbd_helpers
        _, key_scene, _, key_shape_scene = tbd_helpers.generate_keys(self.atomspace)
        data = FloatValue(list(features.numpy().flatten()))
        boundingBoxInstance = self.atomspace.add_node(types.ConceptNode, 'BoundingBox1')
        boundingBoxInstance.set_value(key_scene, data)
        boundingBoxInstance.set_value(key_shape_scene, FloatValue(list(features.numpy().shape)))
        box_concept = self.atomspace.add_node(types.ConceptNode, 'BoundingBox')
        self.atomspace.add_link(types.InheritanceLink, [boundingBoxInstance, box_concept])


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
    parser.add_argument('--use-pm', dest='use_pm', action='store_true',
                        help='use pattern matcher')
    parser.add_argument('--no-use-pm', dest='use_pm', action='store_false',
                        help='use URE instead of pattern matcher')
    parser.set_defaults(use_pm=True)
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
        if args.use_pm:
            atomspace = initialize_atomspace_by_facts(args.atomspaceFileName)
        else:
            scheme_directories = ["~/projects/opencog/examples/pln/conjunction/",
                                  "~/projects/atomspace/examples/rule-engine/rules/",
                                  "~/projects/opencog/opencog/pln/rules/"]

            atomspace = initialize_atomspace_by_facts(args.atomspaceFileName,
                                                      "conjunction-rule-base-config.scm",
                                                      [os.path.expanduser(x) for x in scheme_directories])
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
        pmVqaPipeline.answerQuestionsFromFile(args.questionsFileName, use_pm=args.use_pm)

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
