## Experiments with multiple DNNs for Visual Question Answering (VQA)

The goal of our experiment was to evaluate ability of DNNs batch to be trained to solve VQA task and experiment with different loss functions. 

We trained models in supervised manner on the [VQA v2 dataset](http://www.visualqa.org/download.html) 

### Training on "yes/no" questions

For the beginning we trained models on yes/no questions only, where each question is described only by the following grounded formula: _predadj(A, B).
It means that a question asks: "does an object A have a property B". Thus we've got 12117 training questions and 5796 validation questions.

All key words (A and B) from all questions are putted into vocabulary. The acquired vocabulary contains 1353 words. So we create a DNN for each word in vocabulary and try to train it to predict is there an object or property described by the word. We used pretrained [bottom-up-attention features](https://github.com/peteanderson80/bottom-up-attention) for bounding boxes as input of DNNs. 

Training iteration flow is the following:

* Get a pair of question key words and ground truth answer from the training set.

* Find indices of the key words in vocabulary.

* Run forward a pair of correspondent DNNs, feeding a batch of boundig boxes features. 

* Get a joint probability of A and B for every bounding box by multiplying outputs of DNN pair.

* Compute a loss function value (binary cross entropy) depending on the acquired joint probability and ground truth answer.

* Do backpropagation.

There are two options of the loss function at the moment:

1. In [train_00_pytorch.py](./train_00_pytorch.py) the loss function gets only a maximum value of the predicted joint probability, if the ground truth answer is "yes", or the whole batch of the predicted joint probabilities, if the answer is "no".  
Thus, in the first case we want to do backpropagation considering the corresponded bounding box with maximum response only, while others are wanted to be ignored. In the second case we want to minimse response for the whole batch, since the answer should be "no".

Acquired validation score for this type of loss is 62.8%.


2. In [train_01_pytorch.py](./train_01_pytorch.py) the loss function always considers all predictions. 


## Using OpenCog PatternMatcher to answer questions

### Overview

```pattern_matcher_vqa.py``` answers questions using OpenCog 
PatternMatcher and neural networks for separate words.

(1) Script loads images features by image id and forms an Atomspace with number of ConceptNode instances. Which ConceptNode instance represents a bounding box of the image. Features are placed into bounding box ConceptNode as FloatValue.

(2) Script gets a question and parses it using question2atomeese Java library. Question2atomeese returns PatternMatcher query on Scheme language. Query is executed and PatternMatcher calls GroundingPredicates on bounding box from Atomspace.

(3) Predicate procedure has two arguments. First is bounding box ConceptNode and second is word ConceptNode. Predicate returns truth value which answers question whether word can be recognized on bounding box. Predicate procedure extracts features from bounding box, converts
them into PyTorch tensor and passes to the NN model to calculate the probability the predicate takes true. NN model is queried from the set of models by word.

### Command line arguments

Three arguments are required to run ```pattern_matcher_vqa.py```:

- QUESTIONSFILENAME - file which contains parsed questions in format which is described in [record.py module](../../question2atomeese/record.py); this file can be generated from visualqa.org data using [get_questions.py](../../question2atomeese/get_questions.py); see [README.md](../../question2atomeese/README.md)

- MODELSFILENAME - file which contains pretrained words model; it can be prepared using [train_01_pytorch.py](./train_01_pytorch.py) script

- FEATURESPATH - folder or .zip file which contains features of bounding boxes

```
$ python pattern_matcher_vqa.py -h
usage: pattern_matcher_vqa.py [-h] --questions QUESTIONSFILENAME --models
                              MODELSFILENAME --features FEATURESPATH
                              [--features-prefix FEATURESPREFIX]
                              [--words WORDSFILENAME]
                              [--opencog-log-level {FINE,DEBUG,INFO,ERROR,NONE}]
                              [--python-log-level {INFO,DEBUG,ERROR}]
                              [--question2atomeese-java-library Q2AJARFILENNAME]

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
  --words WORDSFILENAME, -w WORDSFILENAME
                        words vocabulary (required only if old models file
                        format is used
  --opencog-log-level {FINE,DEBUG,INFO,ERROR,NONE}
                        OpenCog logging level
  --python-log-level {INFO,DEBUG,ERROR}
                        Python logging level
  --question2atomeese-java-library Q2AJARFILENNAME
                        path to question2atomeese-<version>.jar
```
