# Prerequisites

Build Link Grammar with Java bindings (see [link-grammar repo](https://github.com/opencog/link-grammar))

Build RelEx (see [relex repo](https://github.com/opencog/relex))

Install ```maven```:
```
sudo apt install maven
```

Python 3 is required to parse questions dataset.

# Building

Add linkgrammar.jar and relex.jar into local maven repository:
```
mvn install:install-file \
	-Dfile=<linkgrammar-jar-folder/linkgrammar.jar> \
	-DgroupId=org.opencog \
	-DartifactId=linkgrammar \
	-Dversion=<linkgrammar.version> \
	-Dpackaging=jar
mvn install:install-file \
	-Dfile=<relex-jar-folder/relex.jar> \
	-DgroupId=org.opencog \
	-DartifactId=relex \
	-Dversion=<relex.version> \
	-Dpackaging=jar
```

Build using maven:
```
mvn compile
```

# Prepare questions dataset

Download question and annotations from [download section](http://www.visualqa.org/download.html) of VQA site.
```
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
./get_questions.py > questions.txt
```

# Running

Run question2atomeese app:
```
RELEX=<path-to-relex-src-dir> ./question2atomeese.sh questions.txt
```
