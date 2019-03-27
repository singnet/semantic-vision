import torch
from cognets import unpack_args, set_value, get_value, TTruthValue
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


def cog_merge_hi_conf_tv(atom, tv):
    old_tv = get_value(atom, tv=True)
    if old_tv.confidence < tv.confidence:
        set_value(atom, tv, tv=True)
        atom.tv = TruthValue(float(tv.mean), float(tv.confidence))


def fuzzy_conjunction_introduction_formula(conj, conj_set):
    atoms = conj_set.out
    args = list(unpack_args(*conj_set.out, tv=True))
    min_s = torch.min(torch.stack(tuple(x.mean for x in args)))
    min_c = torch.min(torch.stack(tuple(x.confidence for x in args)))
    result = TTruthValue(torch.stack([min_s, min_c]))
    cog_merge_hi_conf_tv(conj, result)
    return conj


def precise_modus_ponens_strength_formula(sA, sAB, snotAB):
    return sAB * sA + snotAB * (1 - sA)


def modus_ponens_formula(B, AB, A):
    sA = get_value(A, tv=True).mean
    cA = get_value(A, tv=True).confidence
    sAB = get_value(AB, tv=True).mean
    cAB = get_value(AB, tv=True).confidence
    snotAB = 0.2 # Huge hack
    cnotAB = 1
    new_tv = TTruthValue(precise_modus_ponens_strength_formula(sA, sAB, snotAB),
                min(cAB, cnotAB, cA))
    cog_merge_hi_conf_tv(B, new_tv)
    return B


def gt_zero_confidence(atom):
    tensor_tv = get_value(atom, tv=True)
    result = TruthValue(0 < tensor_tv.confidence, 1)
    return result


def gen_modus_ponens_rule(link_type):
    A = VariableNode("$A")
    B = VariableNode("$B")
    AB = link_type(A, B)
    # todo: scheme implementation has untyped variables
    # but python version fails in such case
    variable_declaration = VariableList(TypedVariableLink(A, TypeNode("ConceptNode")), TypedVariableLink(B, TypeNode("ConceptNode")))
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
    schema = DefinedSchemaNode('modus-ponens')
    DefineLink(schema, gen_modus_ponens_rule(InheritanceLink))
    MemberLink(schema, rule_base)
    return rule_base

