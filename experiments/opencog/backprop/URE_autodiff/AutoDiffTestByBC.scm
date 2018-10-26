(add-to-load-path ".")

(use-modules (opencog))
(use-modules (opencog query))
(use-modules (opencog exec))
(use-modules (opencog rule-engine))


(define (id2 C A)
  (cog-set-tv! C (stv 1 1)))
  ;C)

(define (id4 C A B D)
  (cog-set-tv! C (stv 1 1)))
  ;C)

(DefineLink
     (DefinedSchemaNode "GradientSum")
     (BindLink
      (VariableList
       (VariableNode "$F1")
       (VariableNode "$F2")
       (VariableNode "$EGX")
       (VariableNode "$EGY"))
      (And
       (NumericOutputLink (ConceptNode "Gradient") (NumericOutputLink (ConceptNode "+") (VariableNode "$F1") (VariableNode "$F2")))
       (Evaluation (Predicate "ExpandTo") (List (NumericOutputLink (ConceptNode "Gradient") (Variable "$F1")) (Variable "$EGX")))
       (Evaluation (Predicate "ExpandTo") (List (NumericOutputLink (ConceptNode "Gradient") (Variable "$F2")) (Variable "$EGY"))))

      (ExecutionOutputLink
       (GroundedSchemaNode "scm: id4")
       (List
        (EvaluationLink
         (Predicate "ExpandTo")
         (List
          (NumericOutputLink (ConceptNode "Gradient") (NumericOutputLink (ConceptNode "+") (VariableNode "$F1") (VariableNode "$F2")))
          (NumericOutputLink (ConceptNode "+")
                             (VariableNode "$EGX")
                             (VariableNode "$EGY"))))
        (NumericOutputLink (ConceptNode "Gradient") (NumericOutputLink (ConceptNode "+") (VariableNode "$F1") (VariableNode "$F2")))
        (Evaluation (Predicate "ExpandTo") (List (NumericOutputLink (ConceptNode "Gradient") (Variable "$F1")) (Variable "$EGX")))
        (Evaluation (Predicate "ExpandTo") (List (NumericOutputLink (ConceptNode "Gradient") (Variable "$F2")) (Variable "$EGY")))
        ))))


;(Evaluation (Predicate "ExpandTo") (List (NumericOutputLink (ConceptNode "Gradient") (Number "3")) (Number "-3")))
;(Evaluation (Predicate "ExpandTo") (List (NumericOutputLink (ConceptNode "Gradient") (Number "4")) (Number "-4")))



(DefineLink
     (DefinedSchemaNode "GradientAny")
     (BindLink
      (VariableList
       (TypedVariable (VariableNode "$F0") (Type "NumberNode")))
      (And
       (NumericOutputLink (ConceptNode "Gradient") (Variable "$F0")))
      (ExecutionOutputLink
       (GroundedSchemaNode "scm: id2")
       (List
        (Evaluation (Predicate "ExpandTo") (List (NumericOutputLink (ConceptNode "Gradient") (Variable "$F0"))
                                                 (Number "0")))
        (NumericOutputLink (ConceptNode "Gradient") (Variable "$F0"))
        ))
      ))


(define SRC
  (Evaluation (Predicate "ExpandTo")
              (List
               (NumericOutputLink (ConceptNode "Gradient") (NumericOutputLink (ConceptNode "+") (NumericOutputLink (ConceptNode "+") (NumberNode 1) (NumberNode 2))
                                                                              (NumericOutputLink (ConceptNode "+") (NumberNode 3) (NumberNode 4))))
               (Variable "$X"))))

(define SRC2 (Evaluation (Predicate "ExpandTo")
                         (List (NumericOutputLink (ConceptNode "Gradient") (NumericOutputLink (ConceptNode "+") (Number "4") (Number "3")))
                               (Variable "$X"))))

(define SRC23 (Evaluation (Predicate "ExpandTo")
                         (List (NumericOutputLink (ConceptNode "Gradient") (NumericOutputLink (ConceptNode "+") (Number "2") (Number "3")))
                               (NumericOutputLink (Concept "+") (Number "0") (Variable "$X")))))

(define SRC3 (Evaluation (Predicate "ExpandTo")
                         (List (NumericOutputLink (ConceptNode "Gradient") (Number "3"))
                               (Variable "$X"))))

(ExecutionLink
   (SchemaNode "URE:maximum-iterations")
   (ConceptNode "my-rule-base")
   (NumberNode "1000"))

(Inheritance (Concept "my-rule-base") (Concept "URE"))

(Member (DefinedSchema "GradientSum" (stv 0.4 1)) (Concept "my-rule-base"))
(Member (DefinedSchema "GradientAny" (stv 0.4 1)) (Concept "my-rule-base"))

(define (my-forward-chainer SRC) (cog-fc (Concept "my-rule-base") SRC))
(define (my-backward-chainer SRC) (cog-bc (Concept "my-rule-base") SRC)) ; #:vardecl (Variable "$X")))

;(GetLink (VariableNode "$X"))

;(load "test_gsum")
