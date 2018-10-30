"""
The module contains classes and functions
for working with VQA pipeline from jupyter notebook
"""

from io import BytesIO
import jpype
from feature.image import ImageFeatureExtractor
import ipywidgets
from IPython.display import display
from ipywebrtc import CameraStream, ImageRecorder

from scipy import misc
import numpy


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


def pil2ipyimage(image, height=None, width=None):
    """
    Convert PILLOW image to ipython Image

    Parameters
    ----------
    image : pil.Image
        pillow image object
    height: int
        height to resize the returned image widget
        parameter will keep proportions
    width: int
        height to resize the returned image widget
        parameter will keep proportions
    """

    b = BytesIO()
    image.save(b, format='png')
    data = b.getvalue()
    image_widget = display.Image(data, format='png', height=height, width=width)
    return image_widget


class MainWindow():
    def __init__(self, images, vqa, use_camera=False):
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
        self.width = 400
        self.height = 400
        if use_camera:
            self.camera = CameraStream(constraints=
                          {'facing_mode': 'user',
                           'audio': False,
                           'video': { 'width': 640, 'height': 480 }
                           })
            self.image_recorder = ImageRecorder(stream=self.camera)
            self.image_recorder.recording = True
            self.image_recorder.autosave = False
        else:
            self.camera = None

    def _draw_image(self, bbox=None):
        if self.image_output is None:
            return
        with self.image_output:
            image = Image.fromarray(self.current_image)
            if bbox is not None:
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=None, outline="red")
            display.display(pil2ipyimage(image, height=self.height, width=self.width))

    def _next_image(self, idx):
        img_path = self.images[idx]
        self.current_image = misc.imread(img_path)
        self._draw_image()

    def _handle_camera(self):
        if self.camera:
            self.current_image = numpy.asarray(Image.open(BytesIO(self.image_recorder.image.value)))[:,:,:3]

    def _handle_submit(self, sender):
        self.label_question.value = self.text.value
        self._clear_widgets()
        self._handle_camera()
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
        if self.camera:
            vbox_images = VBox(children=[self.image_output, self.camera, self.image_recorder])
        else:
            vbox_images = VBox(children=[interact])
        display.display(HBox(children=[vbox_images, vbox], layout=hlayout))

