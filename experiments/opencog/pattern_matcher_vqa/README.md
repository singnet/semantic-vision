## Using OpenCog PatternMatcher to answer questions

### Overview

```pattern_matcher_vqa.py``` answers questions using OpenCog 
PatternMatcher and neural networks for separate words.

(1) Script loads images features by image id and forms an Atomspace with number of ConceptNode instances. Which ConceptNode instance represents a bounding box of the image. Features are placed into bounding box ConceptNode as FloatValue.

(2) Script gets a question and parses it using question2atomese Java library. Question2atomese returns PatternMatcher query on Scheme language. Query is executed and PatternMatcher calls GroundingPredicates on bounding box from Atomspace.

(3) Predicate procedure has two arguments. First is bounding box ConceptNode and second is word ConceptNode. Predicate returns truth value which answers question whether word can be recognized on bounding box. Predicate procedure extracts features from bounding box, converts
them into PyTorch tensor and passes to the NN model to calculate the probability the predicate takes true. NN model is queried from the set of models by word.

### Command line arguments

Three arguments are required to run ```pattern_matcher_vqa.py```:

- QUESTIONSFILENAME - file which contains parsed questions in format which is described in [record.py module](../../question2atomese/record.py); this file can be generated from visualqa.org data using [get_questions.py](../../question2atomese/get_questions.py); see [README.md](../../question2atomese/README.md)

- MODELSFILENAME - file which contains pretrained words model; it can be prepared using [train_01_pytorch.py](./train_01_pytorch.py) script

- FEATURESPATH - folder or .zip file which contains features of bounding boxes

One optional argument is required to answer complex questions:

- ATOMSPACEFILENAME - Scheme program to fill initial Atomspace; thi program can be generated using [question2atomese.sh](../../question2atomese/question2atomese.sh); see [README.md](../../question2atomese/README.md)
```
$ python DNNs/vqa_multi_dnn/pattern_matcher_vqa.py --help
usage: pattern_matcher_vqa.py [-h] --questions QUESTIONSFILENAME --models
                              MODELSFILENAME --features FEATURESPATH
                              [--features-prefix FEATURESPREFIX] --atomspace
                              ATOMSPACEFILENAME
                              [--opencog-log-level {FINE,DEBUG,INFO,ERROR,NONE}]
                              [--python-log-level {INFO,DEBUG,ERROR}]
                              [--question2atomese-java-library Q2AJARFILENNAME]

Load pretrained words models and answer questions using OpenCog PatternMatcher

optional arguments:
  -h, --help            show this help message and exit
  --questions QUESTIONSFILENAME, -q QUESTIONSFILENAME
                        parsed questions file name
  --models MODELSFILENAME, -m MODELSFILENAME
                        models file name
  --features FEATURESPATH, -f FEATURESPATH
                        features path (it can be either zip archive or folder
                        name)
  --features-prefix FEATURESPREFIX
                        features prefix to be merged with path to open feature
  --atomspace ATOMSPACEFILENAME, -a ATOMSPACEFILENAME
                        Scheme program to fill atomspace with facts
  --opencog-log-level {FINE,DEBUG,INFO,ERROR,NONE}
                        OpenCog logging level
  --python-log-level {INFO,DEBUG,ERROR}
                        Python logging level
  --question2atomese-java-library Q2AJARFILENNAME
                        path to question2atomese-<version>.jar
```
