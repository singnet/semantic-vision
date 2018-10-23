## Experiments with URE on example of automatic expansion of differentiation rules (sum and product rules)  

There are several experiments on implementation of automatic distribution of differentiation rules (sum and product) with URE

* [AutoDiffTest.scm](AutoDiffTest.scm)  - "naive" example using forward chainer for parsing source expressions. Sourece expressions represented through ListLinks.  It's not working properly due to inability of FC to recursively apply basic rules to intermediate inference results. Problem arised on a stage where Pattern Matcher meets restrictions for matching the ListLinks. 
* [AutoDiffTestByPM.scm](AutoDiffTestByPM.scm)  - extented version of AutoDiffTest example. All the ListLinks in source expressions (to be parsed) are replaced by NumericOutputLinks. This example does one-step reasoning inference and doesn't go further. PatternMatcher is used for convertion of forward chainer's outputs to convenient form of PlusLinks and TimesLinks.
* [GradientUnitTest.scm](GradientUnitTest.scm)  - valid version of AutoDiff example implemented using backward chainer. Rules and source expressions are described by NumericOutputLinks. Distribution of differentiation rules is done by finding such distributed result, which can be backtracked to the initial expression. 
* [AutoDiffTestByBC.scm](AutoDiffTestByBC.scm)  -  AutoDiff example implemented using backward chainer. Not working with SRC2 expression due to incorrect assignment of premises.

