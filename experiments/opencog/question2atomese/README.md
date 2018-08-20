# Overview

This folder contains Java code to parse questions in natural language and convert them into OpenCog Pattern Matcher queries in Scheme programming language.
Primary project design goal is to parse [VQA questions dataset](http://www.visualqa.org) and use it in [Pattern Matcher VQA pipeline](../pattern_matcher_vqa), so it also includes scripts to convert VQA data into internal format.
Question2Atomese uses [OpenCog RelEx library](https://github.com/opencog/relex) for natural language processing and uses its results.
RelEx in turn uses [Link Grammar library](https://github.com/opencog/link-grammar), so the full pipeline is:

```
"Is the room dark?" 
    -(LG)-> (S is.v (NP the room.s) (ADJP dark.a) ?)
    -(RelEx)-> _predadj(room, dark)
    -(Question2Atomese)-> (SatisfactionLink ...)
```

Please see [RelEx wiki pages](https://wiki.opencog.org/w/Dependency_relations) for RelEx notation explanation.

# Supported types of the questions

## Sole predicative adjectival modifier YES/NO question (_predadj(object, predicate))

Example: Is the room dark?

Link Grammar parsing results:
```
(S is.v (NP the room.s) (ADJP dark.a) ?)

    +-------------Xp-------------+
    |       +-------Pa-------+   |
    |       +--SIs*x--+      |   |
    +-->Qd--+   +Ds**c+      |   +--RW--+
    |       |   |     |      |   |      |
LEFT-WALL is.v the room.s dark.a ? RIGHT-WALL
```

RelEx parsing results:
```
Dependency relations:

    _predadj(room, dark)

Attributes:

    pos(?, punctuation)
    pos(be, verb)
    noun_number(room, singular)
    definite-FLAG(room, T)
    pos(room, noun)
    pos(the, det)
    TRUTH-QUERY-FLAG(dark, T)
    pos(dark, adj)
    tense(dark, present)
    HYP(dark, T)
```

OpenCog Pattern Matcher query in Scheme:
```
(SatisfactionLink
  (TypedVariableLink (VariableNode "$X") (TypeNode "ConceptNode"))
  (AndLink
    (InheritanceLink (VariableNode "$X") (ConceptNode "BoundingBox"))
    (EvaluationLink (GroundedPredicateNode "py:runNeuralNetwork") (ListLink (VariableNode "$X") (ConceptNode "room")) )
    (EvaluationLink (GroundedPredicateNode "py:runNeuralNetwork") (ListLink (VariableNode "$X") (ConceptNode "dark")) )
  )
)
```

## Subject + object + subject determiner question (_det(subj, ?);_obj(verb, obj);_subj(verb, subj))

Example: What color is the plane?

Link Grammar parsing results:
```
(S what color.s (VP is.v (NP the plane.n)) ?)



    +-----------------Xp----------------+
    +-------->WV-------->+---Ost---+    |
    +-->Ws--+Ds*wc+-Ss*s-+   +Ds**c+    +--RW--+
    |       |     |      |   |     |    |      |
LEFT-WALL what color.s is.v the plane.n ? RIGHT-WALL
```

Relex parsing results:
```
Dependency relations:

    _det(color, _$qVar)
    _obj(be, plane)
    _subj(be, color)

Attributes:

    QUERY-FLAG(color, T)
    noun_number(color, singular)
    pos(color, noun)
    pos(_$qVar, det)
    QUERY-TYPE(_$qVar, what)
    pos(?, punctuation)
    noun_number(plane, singular)
    definite-FLAG(plane, T)
    pos(plane, noun)
    pos(the, det)
    pos(be, verb)
    tense(be, present)
```

OpenCog Pattern Matcher query in Scheme:
```
(BindLink
  (VariableList
    (TypedVariableLink (VariableNode "$B") (TypeNode "ConceptNode"))
    (TypedVariableLink (VariableNode "$X") (TypeNode "ConceptNode"))
  )
  (AndLink
    (InheritanceLink (VariableNode "$B") (ConceptNode "BoundingBox"))
    (InheritanceLink (VariableNode "$X") (ConceptNode "color"))
    (EvaluationLink (GroundedPredicateNode "py:runNeuralNetwork") (ListLink (VariableNode "$B") (ConceptNode "plane")) )
    (EvaluationLink (GroundedPredicateNode "py:runNeuralNetwork") (ListLink (VariableNode "$B") (VariableNode "$X")) )
  )
  (ListLink (Variable "$B") (Variable "$X") (ConceptNode "plane"))
)
```

# Further steps

30% VQA questions database contains only 10 question patterns, which are shown in table below:

| # of questions matched | type of question | formula                                       |
|------------------------|------------------|-----------------------------------------------|
| 27950                  | other            | _det(A, B);_obj(C, D);_subj(C, A)             |
| 13963                  | yes/no           | _predadj(A, B)                                |
| 11179                  | number           | _predadj(A, B);_quantity(A, C)                |
| 10950                  | yes/no           | _obj(A, B);_subj(A, C)                        |
| 10546                  | other            | _obj(A, B);_subj(A, C)                        |
| 8950                   | number           | _pobj(B, D);_quantity(A, C);_predadj(A, B)    |
| 8738                   | other            | _amod(A, B);_subj(C, D);_obj(C, A)            |
| 8508                   | other            | _pobj(A, B);_predadj(C, A)                    |
| 7256                   | yes/no           | _amod(A, B);_subj(C, D);_obj(C, A)            |
| 6510                   | other            | _det(A, B);_poss(C, D);_obj(E, C);_subj(E, A) |

Table above can be generated using ```sort_questions_by_frequency.sh``` script.
New converter to be added to cover more types of questions.

# Usage guide

## Prerequisites

Build Link Grammar with Java bindings (see 
[link-grammar repo](https://github.com/opencog/link-grammar))

Install ```maven```:
```
sudo apt install maven
```

Build RelEx using maven (see ["With Maven" section of RelEx README.md](https://github.com/opencog/relex#with-maven))

Python 3 is required to parse questions dataset. 
Python libraries:
- ```ijson``` - JSON streaming parsing library

## Building

Build using maven:
```
mvn package
```

## Prepare questions dataset

Download question and annotations from 
[download section](http://www.visualqa.org/download.html) of VQA site.
```
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
python get_questions.py -q v2_OpenEnded_mscoco_train2014_questions.json 
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

## Parse questions using RelEx

Run question2atomese app:
```
./question2atomese.sh -i questions.txt -o parsed_questions.txt -a atomspace.scm
```

question2atomese usage:
```
$ ./question2atomese.sh --help
usage: QuestionToOpencogApp
 -a,--atomspace <arg>   filename for atomspace which is calculated from
                        questions
 -i,--input <arg>       input filename, stdin if not provided
 -o,--output <arg>      output filename, stdout if not provided
 ```

## Sort question types by frequency

Get 10 most frequent question types:
```
./sort_questions_by_frequency.sh parsed_questions.txt | head -10
```

## Other scripts

- ```record.py``` - reusable module to load question record from Python 
- ```get_words.py``` - get key words from parsed questions file
- ```unique_questions.py``` - calculate number of questions with unique words in validation dataset
- ```generate_training_questions.py``` - straightforward yes/no questions generator which uses _det(A, B) questions as input
