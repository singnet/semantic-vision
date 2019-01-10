import io
import imageio
import threading
from concurrent import futures

import jpype
from feature.image import ImageFeatureExtractor
from scipy import misc
from splitnet.splitmultidnnmodel import SplitMultidnnRunner
from util import initialize_atomspace_by_facts
from pattern_matcher_vqa import PatternMatcherVqaPipeline, runNeuralNetwork
import network_runner
import grpc
import time

import service_pb2
import service_pb2_grpc



question2atomeseLibraryPath = ('../question2atomese/target/question2atomese-1.0-SNAPSHOT.jar')


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

mydata = threading.local()

class VqaService(service_pb2_grpc.VqaServiceServicer):
   def __init__(self):
       super().__init__()

   @property
   def vqa(self):
       # jpype is not threadsafe, causes segmentation fault
       # if run from another thread
       if getattr(mydata, 'vqa', None) is None:
           mydata.vqa = build_vqa()
       return mydata.vqa

   def answer(self, request, context):
       image = imageio.imread(io.BytesIO(request.data))
       question = request.question
       response = service_pb2.VqaResponse()
       response.ok = False
       try:
           answer = self.vqa.answerQuestionByImage(image, question, use_pm=request.use_pm)
           if answer:
               response.message = answer.answer
               response.ok = True
           else:
               response.message = ''
       except RuntimeError as e:
           print(e)
           return response
       response.ok = True
       return response


def main():
    image = imageio.imread('/home/relex/projects/data/coco/COCO_val2014_000000999999.jpg')
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
