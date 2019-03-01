import torch
from module import unpack_args, set_value, get_value
import opencog
import opencog.atomspace
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.bindlink import execute_atom


def generate_conjunction_rule(nary):
    var_list_typed = VariableList(*[
        TypedVariableLink(
             VariableNode("$X-{0}".format(i)),
             TypeChoice(
                TypeNode("EvaluationLink"),
                TypeNode("InheritanceLink"),
                TypeNode("OrLink"),
                TypeNode("NotLink"),
                TypeNode("ExecutionLink"))) for i in range(nary)])
    var_list = [VariableNode("$X-{0}".format(i)) for i in range(nary)]
    condition = AndLink(*var_list)
    conclusion = ExecutionOutputLink(
        GroundedSchemaNode( "py: pln.fuzzy_conjunction_introduction_formula"),
        ListLink(AndLink(*var_list), SetLink(*var_list)))
    result = BindLink(var_list_typed, condition, conclusion)
    return result


def fuzzy_conjunction_introduction_formula(conj, conj_set):
    atoms = conj_set.out
    args = list(unpack_args(*conj_set.out))
    result = torch.min(*args)
    set_value(conj, result)
    conj.tv = TruthValue(float(result), 1.0)
    return conj


def precise_modus_ponens_strength_formula(sA, sAB, snotAB):
    return sAB * sA + snotAB * (1 - sA)


def modus_ponens_formula(B, AB, A):
    sA = get_value(A)[MEAN]
    cA = get_value(A)[CONFIDENCE]
    sAB = get_value(AB)[MEAN]
    cAB = get_value(AB)[CONFIDENCE]
    snotAB = 0.2 # Huge hack
    cnotAB = 1
    B.tv = TruthValue(precise_modus_ponens_strength_formula(sA, sAB, snotAB),
                   min(min(cAB, cnotAB), cA))
    return B


def gt_zero_confidence(atom):
    tensor_tv = get_value(atom)
    return TruthValue(0 < tensor_tv, 1)


def gen_modus_ponens_rule(link_type):
    A = VariableNode("$A")
    B = VariableNode("$B")
    AB = link_type(A, B)
    variable_declaration = VariableList(A, B)
    patterns = AndLink(
          # Preconditions
          EvaluationLink(
            GroundedPredicateNode( "py: pln.gt_zero_confidence"), A),
          EvaluationLink(
              GroundedPredicateNode( "py: pln.gt_zero_confidence"), AB),
          # Pattern clauses
          AB,
          A)
    rewrite = ExecutionOutputLink(
      GroundedSchemaNode( "py: pln.modus_ponens_formula"),
      ListLink(B, AB, A))
    result = BindLink(variable_declaration, patterns, rewrite)
    return result


def initialize_pln():
    rule_base = ConceptNode('PLN')
    for i in range(1, 3):
        schema = DefinedSchemaNode('fuzzy-conjuntion-rule-{0}'.format(i))
        DefineLink(schema, generate_conjunction_rule(i))
        MemberLink(schema, rule_base)
#    modus-ponens doesn't work yet due to generation of links during inference process
#    schema = DefinedSchemaNode('modus-ponens')
#    DefineLink(schema, gen_modus_ponens_rule(InheritanceLink))
#    MemberLink(schema, rule_base)
    return rule_base

