# Proof of concept for OpenCog API to query and train neural networks

## Description

Script contains examples of GroundedSchemaNodes which can be used to query and train NNs from OpenCog.

For instance system should answer question 'Is the hare grey?' by picture. Then OpenCog boolean formula in atomese:

```
(AndLink
  (ConceptNode "hare.nn")
  (ConceptNode "grey.nn")
)
```

To evaluate it use:
```
(ExecutionOutputLink
  (GroundedSchemaNode "py:runNeuralNetwork")
  (ListLink
    (AndLink
      (ConceptNode "hare.nn")
      (ConceptNode "grey.nn")
    )
    (ConceptNode "boundingBox-1")
  )
)
```

To train it use:
```
(EvaluationLink
  (GroundedPredicateNode "trainNeuralNetwork")
  (ListLink
    (AndLink
      (ConceptNode "hare.nn")
      (ConceptNode "grey.nn")
    )
    (ConceptNode "predictedValueContainer")
    (ConceptNode "expectedValueContainer")
  )
)
```

Predicted value and expected value ConceptNodes represents PyTorch tensors.

## Issues

Problems:
- there is no Value type in OpenCog to represent tensors
- Values cannot be passed as GroundedPredicateNode and GroundedSchemaNode arguments

