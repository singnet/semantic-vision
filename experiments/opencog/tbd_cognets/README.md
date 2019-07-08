
**TBD**
Visual reasoning for opencog based on "Transparency by Design: Closing the Gap Between Performance and Interpretability in Visual Reasoning" paper.
Requires original model: https://github.com/davidmascharka/tbd-nets.
Cognets - https://github.com/singnet/semantic-vision/tree/master/experiments/opencog/cog_module


Conversion of programs to atomspace queries.  

Conversion code is placed in tbd_helpers.py.
Before conversion can take place, the atomspace should be populated with background
knowledge about modules and what they perform.
Module names are used to create a number of facts in form of concept nodes and inheritance links:  
For example filter_color[red] is used to create this link in the atomspace.

``` 
(InheritanceLink
  (ConceptNode "red") 
  (ConceptNode "color")
)
```

Also for each filter we create EvaluationLink to describe what it filters e.g:

```
(EvaluationLink
  (PredicateNode "filters")
  (ListLink
    (ConceptNode "red")
    (ConceptNode "filter_color[red]")
  ) 
)

``` 

This steps implemented in functions **generate_wrappers** and **setup_inheritance**.
