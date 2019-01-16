#!/usr/bin/env python3

import io
import imageio
import logging
import threading
import sys
from concurrent import futures

import jpype
from feature.image import ImageFeatureExtractor
from splitnet.splitmultidnnmodel import SplitMultidnnRunner
from util import initialize_atomspace_by_facts
from pattern_matcher_vqa import PatternMatcherVqaPipeline, runNeuralNetwork
import network_runner
import grpc
import time
import vqaservice
sys.path.append(vqaservice.__path__[0])


from vqaservice import service_pb2, service_pb2_grpc

logger = logging.getLogger(__name__)
question2atomeseLibraryPath = ('../question2atomese/target/question2atomese-1.0-SNAPSHOT.jar')


def setup_logger():
    logging.basicConfig(filename='vqa_service.log', level=logging.INFO)


def build_vqa():
    prototxt = '/home/relex/projects/data/test.prototxt'
    caffemodel = '/home/relex/projects/data/resnet101_faster_rcnn_final_iter_320000_for_36_bboxes.caffemodel'
    models = '/home/relex/projects/data/visual_genome/'
    atomspace_path = '/home/relex/projects/data/train_tv_atomspace.scm'
    jpype.startJVM(jpype.getDefaultJVMPath(),
                   '-Djava.class.path=' + question2atomeseLibraryPath)

    question_converter = jpype.JClass('org.opencog.vqa.relex.QuestionToOpencogConverter')()
    extractor = ImageFeatureExtractor(prototxt, caffemodel)

    scheme_directories = ["/home/relex/projects/opencog/examples/pln/conjunction/",
                          "/home/relex/projects/atomspace/examples/rule-engine/rules/",
                          "/home/relex/projects/opencog/opencog/pln/rules/"]

    atomspace = initialize_atomspace_by_facts(atomspace_path,
                                              "conjunction-rule-base-config.scm",
                                              scheme_directories)
    network_runner.runner = SplitMultidnnRunner(models)
    vqa = PatternMatcherVqaPipeline(extractor, question_converter, atomspace, None)
    return vqa


thread_local = threading.local()


class VqaService(service_pb2_grpc.VqaServiceServicer):

    @property
    def vqa(self):
        # jpype is not threadsafe, causes segmentation fault
        # if run from another thread
        if getattr(thread_local, 'vqa', None) is None:
            thread_local.vqa = build_vqa()
        return thread_local.vqa

    def answer(self, request, context):
        image = imageio.imread(io.BytesIO(request.image_data))
        question = request.question
        response = service_pb2.VqaResponse()
        response.ok = False
        try:
            answer = self.vqa.answerQuestionByImage(image, question, use_pm=request.use_pm)
            if answer.ok:
                response.answer = answer.answer
                response.ok = True
            else:
                response.error_message = answer.error_message
        except RuntimeError as e:
            logger.error(e)
            response.error_message = str(e)
        logger.info(response)
        return response


def main():
    setup_logger()
    service = VqaService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    service_pb2_grpc.add_VqaServiceServicer_to_server(
        service,
        server)
    server.add_insecure_port('127.0.0.1:12345')
    server.start()

    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
