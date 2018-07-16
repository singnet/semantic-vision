# Prerequisites

Build Link Grammar with Java bindings (see 
[link-grammar repo](https://github.com/opencog/link-grammar))

Install ```maven```:
```
sudo apt install maven
```

Build RelEx using maven (see ["With Maven" section of RelEx README.md](https://github.com/opencog/relex#with-maven))

Python 3 is required to parse questions dataset.

# Building

Build using maven:
```
mvn package
```

# Prepare questions dataset

Download question and annotations from 
[download section](http://www.visualqa.org/download.html) of VQA site.
```
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
./get_questions.py -q v2_OpenEnded_mscoco_train2014_questions.json 
	-a v2_mscoco_train2014_annotations.json > questions.txt
```

get_questions.py usage:
```
usage: get_questions.py [-h] --questions QUESTIONSFILENAME
                        [--annotations ANNOTATIONSFILENAME] [--test]
                        [--loglevel {INFO,DEBUG,ERROR}]

Convert set of questions and annotations to plain file with delimiters.

optional arguments:
  -h, --help            show this help message and exit
  --questions QUESTIONSFILENAME, -q QUESTIONSFILENAME
                        questions json filename
  --annotations ANNOTATIONSFILENAME, -a ANNOTATIONSFILENAME
                        annotations json filename
  --test                test mode, process only 10 first questions
  --loglevel {INFO,DEBUG,ERROR}
                        logging level
```

# Parse questions using RelEx

Run question2atomese app:
```
RELEX=<path-to-relex-src-dir> ./question2atomese.sh questions.txt \
	> parsed_questions.txt
```

# Sort question types by frequency

Get 10 most frequent question types:
```
./sort_questions_by_frequency.sh parsed_questions.txt | head -10
```

# Other scripts

- ```get_words.py``` - get key words from parsed questions file
- ```unique_questions.py``` - calculate number of questions with unique words in validation dataset
