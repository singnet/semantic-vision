# VqaMainLoop.py

Script demonstrates PatternMatcher ability to call GroundedPredicateNodes 
which implemented via Python NN. 

(1) Script gets an image gets NN features from the image using 
GetFeatures.getFeatures and forms an Atomspace with only ConceptNode 
representing image as bounding box. Features are placed into bounding box 
as FloatValue.

(2) Script gets a question parses it using question2atomeese Java library. 
question2atomeese returns PatternMatcher query on Scheme language. After 
that the query is executed and PatternMatcher calls GroundingPredicates on 
bounding box from Atomspace.

(3) Each predicate procedure extracts features from bounding box, converts
them into numpy array and pass to the GetFeatures.predicate to calculate the 
probability the predicate takes true.

## Prerequisites

```jpype``` library is used to integrate Python with Java library.

For Ubuntu run:
```
sudo apt-get install python3-jpype
```

question2atomeese Java library should be built.
See [../question2atomeese/README.md](../question2atomeese/README.md)