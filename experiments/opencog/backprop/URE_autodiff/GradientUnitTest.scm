(add-to-load-path ".")

;load opencog modules
(use-modules (opencog))
(use-modules (opencog query))
(use-modules (opencog exec))
(use-modules (opencog rule-engine))

;3 versions of identity function. id2 and id4 return first argument with assigned truth values
(define (id2 C A)
  (cog-set-tv! C (stv 1 1)))

(define (id4 C A B D)
  (cog-set-tv! C (stv 1 1)))

(define (identity A B)
  A)

;auxiliary cutbacks for "global constants"
(define Gradient (ConceptNode "Gradient"))

(define ExpandTo (Predicate "ExpandTo"))

(define Plus (ConceptNode "+"))

(define NumberT (TypeNode "NumberNode"))

(define rbs (ConceptNode "my-rule-base"))

(define GradientSum (DefinedSchemaNode "GradientSum"))

(define GradientAny (DefinedSchemaNode "GradientAny"))

(define N0 (NumberNode "0"))

(define	N3 (NumberNode "3"))

(define N4 (NumberNode "4"))

;auxiliary cutbacks for atomese variables
(define X  (VariableNode "$X"))
(define F0 (VariableNode "$F0"))
(define F1 (VariableNode "$F1"))
(define F2 (VariableNode "$F2"))
(define EGX (VariableNode "$EGX"))
(define EGY (VariableNode "$EGY"))

;predicates for propagating gradient througt an expression. Used as premises for the rule base 
(define premise1 (EvaluationLink ExpandTo (List (NumericOutputLink Gradient F1) EGX)))

(define premise2 (EvaluationLink ExpandTo (List (NumericOutputLink Gradient F2) EGY)))

;encoding of the sum rule in differentiation. Assuming just two elements in summation
(DefineLink GradientSum
            (BindLink
             (VariableList
              F1
              F2
              EGX
              EGY)
             (And premise1 premise2)
             (ExecutionOutputLink
              (GroundedSchemaNode "scm: id4")
              (List
               (EvaluationLink
                ExpandTo
                (List (NumericOutputLink Gradient (NumericOutputLink Plus F1 F2))
                      (NumericOutputLink Plus EGX EGY)))
                premise1 premise2)
              )
             )
            )

;auxiliary term for defining rule of differentiation of constant (scalar) 
(define GradF0 (NumericOutputLink Gradient F0))

;trivial rule for  differentiation of constant (scalar)
(DefineLink GradientAny
            (BindLink
             (TypedVariableLink F0 NumberT)
             F0
             (ExecutionOutputLink
              (GroundedSchemaNode "scm: identity")
              (List
               (EvaluationLink
                ExpandTo
                (List GradF0 N0))
               ))
             ))

;expression to be "differentiated"
(define SRC2 (Evaluation (Predicate "ExpandTo")
                         (List (NumericOutputLink (ConceptNode "Gradient") (NumericOutputLink (ConceptNode "+") (Number "4") (Number "3")))
                               (Variable "$X"))))

;initialization of URE
(ExecutionLink
   (SchemaNode "URE:maximum-iterations")
   (ConceptNode "my-rule-base")
   (NumberNode "1000"))

(Inheritance (Concept "my-rule-base") (Concept "URE"))

(Member (DefinedSchema "GradientSum" (stv 0.4 1)) (Concept "my-rule-base"))
(Member (DefinedSchema "GradientAny" (stv 0.4 1)) (Concept "my-rule-base"))

;definig alias for convenience
(define (my-backward-chainer SRC) (cog-bc (Concept "my-rule-base") SRC))

;definig expected output
(define expected (SetLink
                  (Evaluation
                   ExpandTo
                   (List
                    (NumericOutputLink
                     Gradient
                     (NumericOutputLink
                      Plus N4 N3))
                    (NumericOutputLink Plus N0 N0)))))

;recursively compare results of backward-chainer and expected output
(equal? (my-backward-chainer SRC2) expected)