(add-to-load-path ".")

(use-modules (opencog))
(use-modules (opencog query))
(use-modules (opencog exec))
(use-modules (opencog rule-engine))


(DefineLink
     (DefinedSchemaNode "GradientMul")    
     (BindLink
      (VariableList
       (VariableNode "$F1")
       (VariableNode "$F2"))
      (AndLink
      (NumericOutputLink (ConceptNode "Gradient") (NumericOutputLink (ConceptNode "*") (VariableNode "$F1") (VariableNode "$F2"))))

      (NumericOutputLink (ConceptNode "+")
		(NumericOutputLink (ConceptNode "*") (VariableNode "$F2") 
                                            (NumericOutputLink (ConceptNode "Gradient") (VariableNode "$F1")))
                (NumericOutputLink (ConceptNode "*") (VariableNode "$F1") 
                                            (NumericOutputLink (ConceptNode "Gradient") (VariableNode "$F2")))
      )
      
      ))


(define SRC (NumericOutputLink (ConceptNode "Gradient") (NumericOutputLink (ConceptNode "*") (NumericOutputLink (ConceptNode "+") (NumberNode 1) (NumberNode 2))
          (NumericOutputLink (ConceptNode "*") (NumberNode 3) (NumberNode 4)))) )
;(define SRC (ListLink (ConceptNode "Gradient") (ListLink (ConceptNode "*") (NumberNode 1) (NumberNode 2))))

(ExecutionLink
   (SchemaNode "URE:maximum-iterations")
   (ConceptNode "my-rule-base")
   (NumberNode "10000")
   )

(Inheritance (Concept "my-rule-base") (Concept "URE"))

(Member (DefinedSchema "GradientMul" (stv 0.4 1)) (Concept "my-rule-base"))

(define (my-forward-chainer SRC) (cog-fc (Concept "my-rule-base") SRC))

(GetLink (VariableNode "$X"))

(Inheritance (Concept "my-rule-base") (Concept "URE"))

(Member (DefinedSchema "GradientMul" (stv 0.4 1)) (Concept "my-rule-base"))

(define (my-forward-chainer SRC) (cog-fc (Concept "my-rule-base") SRC))

(GetLink (VariableNode "$X"))

