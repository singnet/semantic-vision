"""
The module contains classes and functions
for working with VQA pipeline from jupyter notebook
"""

import jpype
from feature.image import ImageFeatureExtractor
from ipywidgets import Image
from IPython.display import display

from scipy import misc


from ipywidgets import Layout
from ipywidgets import VBox, HBox
from ipywidgets import widgets


from IPython import display
from PIL import Image, ImageDraw


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
        self.image_output = None
        self.label_question = widgets.Label(value="Is the plane red?")
        self.label_answer = widgets.Label(value="")
        self.label_query = widgets.Output(layout={'border': 'none'})
        self.label_expression = widgets.Output(layout={'border': 'none'})
        self.current_image = None
        self.text = widgets.Text()
        self.text.on_submit(self._handle_submit)
        self.use_pattern_matcher = True

    def _draw_image(self, bbox=None):
        if self.image_output is None:
            return
        with self.image_output:
            image = Image.fromarray(self.current_image)
            if bbox is not None:
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=None, outline="red")
            display.display(image)

    def _next_image(self, idx):
        img_path = self.images[idx]
        self.current_image = misc.imread(img_path)
        self._draw_image()

    def _handle_submit(self, sender):
        self.label_question.value = self.text.value
        self._clear_widgets()
        new_answer = self.vqa.answerQuestionByImage(self.current_image, self.text.value,
                                                    use_pm=self.use_pattern_matcher)
        if not (new_answer and new_answer.answer):
            self.label_answer.value = "I don't know."
            return
        else:
            self.label_answer.value = new_answer.answer
            with self.label_query:
                print(new_answer.query)
            with self.label_expression:
                print(new_answer.answerExpression)
        self.image_output.clear_output()
        if new_answer.answerBox is not None:
            self._draw_image(bbox=new_answer.boundingBoxes[new_answer.answerBox])
        else:
            self._draw_image()

    def _clear_widgets(self):
        self.label_answer.value = ""
        self.label_query.clear_output()
        self.label_expression.clear_output()

    def display(self):
        hlayout = Layout(display='flex',
                         flex_flow='row',
                         align_items='stretch',
                         border='none',
                         width='100%')

        vbox = VBox(children=[self.text, self.label_answer, self.label_query, self.label_expression])
        vbox.layout.width = '70%'
        interact = widgets.interactive(self._next_image, idx=(0, len(self.images) - 1))
        interact.layout.height = '550px'
        self.image_output = interact.out
        display.display(HBox(children=[interact, vbox], layout=hlayout))
