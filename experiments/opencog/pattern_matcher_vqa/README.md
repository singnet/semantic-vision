## Using OpenCog PatternMatcher to answer questions

### Overview

```pattern_matcher_vqa.py``` answers questions using OpenCog 
PatternMatcher and neural networks for separate words.

(1) Script loads images features by image id and forms an Atomspace with number of ConceptNode instances. Which ConceptNode instance represents a bounding box of the image. Features are placed into bounding box ConceptNode as FloatValue.

(2) Script gets a question and parses it using question2atomese Java library. Question2atomese returns PatternMatcher query on Scheme language. Query is executed and PatternMatcher calls GroundingPredicates on bounding box from Atomspace.

(3) Predicate procedure has two arguments. First is bounding box ConceptNode and second is word ConceptNode. Predicate returns truth value which answers question whether word can be recognized on bounding box. Predicate procedure extracts features from bounding box, converts
them into PyTorch tensor and passes to the NN model to calculate the probability the predicate takes true. NN model is queried from the set of models by word.

### Command line arguments

The following arguments are required to run ```pattern_matcher_vqa.py```:

--model-kind - two kinds of models are supported: "Multi DNN" and "Hypernet"
- QUESTIONSFILENAME - file which contains parsed questions in format which is described in [record.py module](../question2atomese/record.py); this file can be generated from visualqa.org data using [get_questions.py](../question2atomese/get_questions.py); see [README.md](../question2atomese/README.md)
--features-extractor-kind - two kinds of features extractors are supported:
 PRECALCULATED (loading precalculated features from file) and IMAGE (extracting features from image)

Required for "Multi DNN" model:
- MULTIDNNMODELFILENAME - pretrained "Multi DNN" model file

Required for "Hypernet" model:
- HYPERNETMODELFILENAME - pretrained "Hypernet" model file
- HYPERNETWORDSFILENAME - file which contains words dictionary; required for "Hypernet" model only; "Multi DNN" contains dictionary in model file
- HYPERNETWORDEMBEDDINGSFILENAME - file which contains words embeddings model; required for "Hypernet" model only; "Multi DNN" doesn't use word embeddings

Required for precalculated feature extractor:
- FEATURESPATH - folder or .zip file which contains features of bounding boxes
- PRECALCULATEDFEATURESPREFIX - file prefix to merge with image id and FEATURESPATH to get full file name

Required for image feature extractor:
- IMAGESPATH - folder or .zip file which contains images
- IMAGESPREFIX - file prefix to merge with image id and IMAGESPATH to get full file name
(Bottom-up-attention libraries)[https://github.com/peteanderson80/bottom-up-attention.git] should be built and added into LD_LIBRARY_PATH and PYTHONPATH. See [./feature/README.md](./feature/README.md) for instructions.

One optional argument is required to answer complex questions:
- ATOMSPACEFILENAME - Scheme program to fill initial Atomspace; thi program can be generated using [question2atomese.sh](../question2atomese/question2atomese.sh); see [README.md](../question2atomese/README.md)

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
```
