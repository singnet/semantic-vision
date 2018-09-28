# setup pipeline
import jpype
from feature.image import ImageFeatureExtractor
from ipywidgets import Image
from scipy import misc
from IPython.display import display
from ipywidgets import Layout

from ipywidgets import VBox, HBox

from ipywidgets import widgets


def build_image_feature_extractor(prototxt, caffemodel):
    return ImageFeatureExtractor(prototxt, caffemodel)


def build_question_to_query_converter(question2atomeseLibraryPath='../question2atomese/target/question2atomese-1.0-SNAPSHOT.jar'):
    jpype.startJVM(jpype.getDefaultJVMPath(),
                   '-Djava.class.path=' + question2atomeseLibraryPath)

    question_converter = jpype.JClass('org.opencog.vqa.relex.QuestionToOpencogConverter')()
    return question_converter


class MainWindow():
    def __init__(self, images, vqa):
        self.vqa = vqa
        self.images = images
        self.label_question = widgets.Label(value="Is the plane red?")
        self.label_answer = widgets.Label(value="")
        self.current_image = None
        self.text = widgets.Text()
        self.text.on_submit(self._handle_submit)

    def _next_image(self, idx):
        img_path = self.images[idx]
        image = Image.from_file(img_path,
                                width=300, height=400)

        self.current_image = misc.imread(img_path)
        image.widght = 320
        image.height = 420
        display(image)

    def _handle_submit(self, sender):
        self.label_question.value = self.text.value
        self.label_answer.value = ""
        new_answer = self.vqa.answerQuestionByImage(self.current_image, self.text.value)
        if new_answer is None:
            self.label_answer.value = "I don't know."
        else:
            self.label_answer.value = new_answer

    def display(self):
        hlayout = Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    border='none',
                    width='100%')


        vbox = VBox(children=[self.text, self.label_answer])
        interact = widgets.interactive(self._next_image, idx=(0, len(self.images) - 1))
        interact.layout.height = '550px'
        display(HBox(children=[interact, vbox], layout=hlayout))


