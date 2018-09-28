import jpype
from feature.image import ImageFeatureExtractor
from scipy import misc
from splitnet.splitmultidnnmodel import SplitMultidnnRunner
from util import initialize_atomspace_by_facts
from pattern_matcher_vqa import PatternMatcherVqaPipeline, runNeuralNetwork
import network_runner


question2atomeseLibraryPath = ('../question2atomese/target/question2atomese-1.0-SNAPSHOT.jar')

def test():
    prototxt = '/mnt/fileserver/shared/vital/image-features/test.prototxt'
    caffemodel = '/mnt/fileserver/shared/vital/image-features/resnet101_faster_rcnn_final_iter_320000_for_36_bboxes.caffemodel'
    models = '/mnt/fileserver/shared/models/vqa_split_multidnn/visual_genome/'
    atomspace_path = '/mnt/fileserver/shared/models/vqa_split_multidnn/vqa_dataset/atomspace_val.scm'
    jpype.startJVM(jpype.getDefaultJVMPath(),
                   '-Djava.class.path=' + question2atomeseLibraryPath)

    question_converter = jpype.JClass('org.opencog.vqa.relex.QuestionToOpencogConverter')()
    extractor = ImageFeatureExtractor(prototxt, caffemodel)

    scheme_directories = ["/home/noskill/projects/opencog/examples/pln/conjunction/",
                          "/home/noskill/projects/atomspace/examples/rule-engine/rules/",
                          "/home/noskill/projects/opencog/opencog/pln/rules/"]

    atomspace = initialize_atomspace_by_facts("atomspace_val.scm",
                                              "conjunction-rule-base-config.scm",
                                              scheme_directories)
    image =  misc.imread('images/Fat-Zebra-Animated-Animal-Photo.jpg')
    network_runner.runner = SplitMultidnnRunner(models)
    vqa = PatternMatcherVqaPipeline(extractor, question_converter, atomspace, None)
    print(vqa.answerQuestionByImage(image, 'Are zebras striped?'))
    print(vqa.answerQuestionByImage(image, 'Are zebras striped?', use_pm=False))

    image =  misc.imread('images/red-plane.jpg')
    print(vqa.answerQuestionByImage(image, 'What color is the plane?'))
    print(vqa.answerQuestionByImage(image, 'What color is the plane?', use_pm=False))

if __name__ == '__main__':
    test()
