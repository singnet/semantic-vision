## Using OpenCog PatternMatcher to answer questions

### Overview

```pattern_matcher_vqa.py``` answers questions using OpenCog 
PatternMatcher and neural networks for separate words.

(1) Script loads images features by image id and forms an Atomspace with number of ConceptNode instances. Which ConceptNode instance represents a bounding box of the image. Features are placed into bounding box ConceptNode as FloatValue.

(2) Script gets a question and parses it using question2atomese Java library. Question2atomese returns PatternMatcher query on Scheme language. Query is executed and PatternMatcher calls GroundingPredicates on bounding box from Atomspace.

(3) Predicate procedure has two arguments. First is bounding box ConceptNode and second is word ConceptNode. Predicate returns truth value which answers question whether word can be recognized on bounding box. Predicate procedure extracts features from bounding box, converts
them into PyTorch tensor and passes to the NN model to calculate the probability the predicate takes true. NN model is queried from the set of models by word.

### Supported types of models and questions

Pattern matcher VQA pipeline supports:

- two kinds of NN models:
  - MULTIDNN - separate NN corresponds to each word, NN inputs only bounding box features
  - HYPERNET - one NN inputs bounding box features and word embedding

- two kinds of image features extractors:
  - PRECALCULATED - precalculated bounding boxes and features for VQA dataset is read from file
  - IMAGE - separate NN is used to extract bounding boxes from image and features for each bounding box. To use this mode (Bottom-up-attention libraries)[https://github.com/peteanderson80/bottom-up-attention.git] should be built and added into LD_LIBRARY_PATH and PYTHONPATH. See [./feature/README.md](https://github.com/singnet/semantic-vision/tree/master/experiments/opencog/pattern_matcher_vqa/feature) for instructions.

User can independently set kind of NN model and kind of features extractor, they can be combined by four different ways.

Pattern matcher VQA pipeline at the moment supports two kinds of questions:
- ```_predadj(A, B)``` questions: for instance "Are the zebras fat?" will be parsed as ```_predadj(zebra, fat)```.
- ```_det(A, B);_obj(C, D);_subj(C, A)```: for instance "What color is the ball?" will be parsed as ```_det(color, _$qVar);_obj(be, ball);_subj(be, color)```. This kind of questions require atomspace database, which contains facts like ```(Inheritance (Concept "red") (Concept "color"))```.

### Usage example

Input questions:
```
$ cat /home/vital/projects/vqa/test_question.txt
11760004::yes/no::Are the zebras fat?::11760::yes::_predadj(A, B)::_predadj(zebra, fat)
527510002::other::What color is the plane?::527510::silver::_det(A, B);_obj(C, D);_subj(C, A)::_det(color, _$qVar);_obj(be, plane);_subj(be, color)
299838003::other::What color is the ball?::299838::yellow::_det(A, B);_obj(C, D);_subj(C, A)::_det(color, _$qVar);_obj(be, ball);_subj(be, color)
```

Executing script:
```
python pattern_matcher_vqa.py \
    --questions /home/vital/projects/vqa/test_question.txt \
    --atomspace /home/vital/projects/vqa/other_det_obj_subj.scm \
    --model-kind MULTIDNN \
    --multidnn-model /home/vital/projects/vqa/model_01_max_score_val.pth.tar \
    --features-extractor-kind PRECALCULATED \
    --precalculated-features /home/vital/projects/vqa/downloaded/val2014_parsed_features.zip \
    --precalculated-features-prefix val2014_parsed_features/COCO_val2014_ \
    --python-log-level INFO \
    --opencog-log-level NONE

11760004::Are the zebras fat?::yes::yes::11760
527510002::What color is the plane?::white::silver::527510
299838003::What color is the ball?::tan::yellow::299838
Questions processed: 3, answered: 3, correct answers: 33.33333333333333% (1)
```

Answers are printed using format: ```questionId::question::answer::correct_answer::imageId``` .

### Datasets and models

Precalculated coco vqa features for validation set, along with parsed questions and  
 multidnn model may be downloaded with download_data.sh

There is also docker image for vqa code - https://hub.docker.com/r/opencog/vqa/

### Main arguments

The following arguments are required to run ```pattern_matcher_vqa.py``` (see full command line parameters description below):

- --model-kind {MULTIDNN,HYPERNET}: set NN model type
- --questions QUESTIONSFILENAME: questions database filename. Questions are kept in files in format described by [record.py](https://github.com/singnet/semantic-vision/blob/master/experiments/opencog/question2atomese/record.py). Main fields which are used by pipeline are ```image_id``` and ```question```. [http://visualqa.org](http://visualqa.org) dataset can be converted to this format using [get_questions.p](https://github.com/singnet/semantic-vision/blob/master/experiments/opencog/question2atomese/get_questions.py) (see [README.md#prepare-questions-dataset](https://github.com/singnet/semantic-vision/blob/master/experiments/opencog/question2atomese/README.md#prepare-questions-dataset))
- --features-extractor-kind {PRECALCULATED,IMAGE}: set kind of features extractor
One optional argument is required to answer complex questions:
- --atomspace ATOMSPACEFILENAME - database of facts to answer ```_det(A, B);_obj(C, D);_subj(C, A)``` questions. It is a Scheme program to fill initial Atomspace; this program can be generated using [question2atomese.sh](https://github.com/singnet/semantic-vision/blob/master/experiments/opencog/question2atomese/question2atomese.sh); see [question2atomese#parse-questions-using-relex](https://github.com/singnet/semantic-vision/tree/master/experiments/opencog/question2atomese#parse-questions-using-relex)

MULTIDNN model parameters:
- --multidnn-model MULTIDNNMODELFILENAME - pretrained "Multi DNN" model file

HYPERNET model parameters:
- --hypernet-model HYPERNETMODELFILENAME - pretrained "Hypernet" model file
- --hypernet-words HYPERNETWORDSFILENAME - file which contains words dictionary; required for "Hypernet" model only; "Multi DNN" contains dictionary in model file
- --hypernet-embeddings HYPERNETWORDEMBEDDINGSFILENAME - file which contains words embeddings model; required for "Hypernet" model only; "Multi DNN" doesn't use word embeddings

PRECALCULATED features extractor parameters:
- --precalculated-features PRECALCULATEDFEATURESPATH - folder or .zip file which contains features of bounding boxes
- --precalculated-features-prefix PRECALCULATEDFEATURESPREFIX - file prefix to merge with image id and FEATURESPATH to get full file name; default is valid for val2014 dataset

IMAGE feature extractor parameters:
- --images IMAGESPATH - folder or .zip file which contains images; it can be downloaded from [visualqa.org](http://images.cocodataset.org/zips/val2014.zip) site
- --images-prefix IMAGESPREFIX - file prefix to merge with image id and IMAGESPATH to get full file name; default is valid for val2014 dataset

### Full list of parameters:
```
$ python pattern_matcher_vqa.py --help
usage: pattern_matcher_vqa.py [-h] --model-kind {MULTIDNN,HYPERNET}
                              --questions QUESTIONSFILENAME
                              [--multidnn-model MULTIDNNMODELFILENAME]
                              [--hypernet-model HYPERNETMODELFILENAME]
                              [--hypernet-words HYPERNETWORDSFILENAME]
                              [--hypernet-embeddings HYPERNETWORDEMBEDDINGSFILENAME]
                              --features-extractor-kind {PRECALCULATED,IMAGE}
                              [--precalculated-features PRECALCULATEDFEATURESPATH]
                              [--precalculated-features-prefix PRECALCULATEDFEATURESPREFIX]
                              [--images IMAGESPATH]
                              [--images-prefix IMAGESPREFIX]
                              [--atomspace ATOMSPACEFILENAME]
                              [--opencog-log-level {FINE,DEBUG,INFO,ERROR,NONE}]
                              [--python-log-level {INFO,DEBUG,ERROR}]
                              [--question2atomese-java-library Q2AJARFILENNAME]

Load pretrained words models and answer questions using OpenCog PatternMatcher

optional arguments:
  -h, --help            show this help message and exit
  --model-kind {MULTIDNN,HYPERNET}, -k {MULTIDNN,HYPERNET}
                        model kind: (1) MULTIDNN requires --model parameter
                        only; (2) HYPERNET requires --model, --words and
                        --embedding parameters
  --questions QUESTIONSFILENAME, -q QUESTIONSFILENAME
                        parsed questions file name
  --multidnn-model MULTIDNNMODELFILENAME
                        Multi DNN model file name
  --hypernet-model HYPERNETMODELFILENAME
                        Hypernet model file name
  --hypernet-words HYPERNETWORDSFILENAME, -w HYPERNETWORDSFILENAME
                        words dictionary
  --hypernet-embeddings HYPERNETWORDEMBEDDINGSFILENAME, -e HYPERNETWORDEMBEDDINGSFILENAME
                        word embeddings
  --features-extractor-kind {PRECALCULATED,IMAGE}
                        features extractor type: (1) PRECALCULATED loads
                        precalculated features; (2) IMAGE extract features
                        from images on the fly
  --precalculated-features PRECALCULATEDFEATURESPATH, -f PRECALCULATEDFEATURESPATH
                        precalculated features path (it can be either zip
                        archive or folder name)
  --precalculated-features-prefix PRECALCULATEDFEATURESPREFIX
                        precalculated features prefix to be merged with path
                        to open feature
  --images IMAGESPATH, -i IMAGESPATH
                        path to images, required only when featur
  --images-prefix IMAGESPREFIX
                        image file prefix to be merged with path to open image
  --atomspace ATOMSPACEFILENAME, -a ATOMSPACEFILENAME
                        Scheme program to fill atomspace with facts
  --opencog-log-level {FINE,DEBUG,INFO,ERROR,NONE}
                        OpenCog logging level
  --python-log-level {INFO,DEBUG,ERROR}
                        Python logging level
  --question2atomese-java-library Q2AJARFILENNAME
                        path to question2atomese-<version>.jar
  --use-pm              use pattern matcher
  --no-use-pm           use URE instead of pattern matcher
```
