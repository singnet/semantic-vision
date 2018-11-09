"""
Converters for tbd programs to atomese and callbacks
"""

import re
import uuid

from opencog.atomspace import types
from opencog.type_constructors import *

import numpy
import torch


relate_reg = re.compile('[a-z]+\[([a-z]+)\]')
filter_reg = re.compile('[a-z]+_([a-z]+)\[([a-z]+)\]')
filter_reg.match('filter_shape[sphere]').groups()
relate_reg.match('relate[front]').groups()
tbd = None


def build_eval_link(atomspace, classify_type, variable, eval_link_sub):
    predicate =  atomspace.add_node(types.GroundedSchemaNode, "py:classify")
    variable = atomspace.add_node(types.VariableNode, "$X")
    lst = atomspace.add_link(types.ListLink, [variable, eval_link_sub])
    return atomspace.add_link(types.EvaluationLink, [predicate, lst])


def build_filter(atomspace, filter_type, filter_argument, exec_out_sub):
    """
    Builds ExecutionOutputLink
    :param atomspace: AtomSpace
    :param filter_type: Atom
        Atom with name = filter type e.g. color or size
    :param filter_argument: Atom
        Atom with name = particulare filter instance e.g. red or large
    :param exec_out_sub: Atom
        Atom which upon execution will have attention map and features attached
    :return: ExecutionOutputLink
    """
    schema = atomspace.add_node(types.GroundedSchemaNode, "py:filter_callback")

    lst = atomspace.add_link(types.ListLink, [filter_type,
                                              filter_argument,
                                              exec_out_sub])
    return atomspace.add_link(types.ExecutionOutputLink, [schema, lst])


def build_relate(atomspace, relate_argument, exec_out_sub):
    """
    Builds ExecutionOutputLink which calls relate callback
    and accepts relation type e.g. "behind"
    :param atomspace:
    :param relate_argument: str
        Relation type name e.g. behind, front
    :param exec_out_sub: Atom
    :return: ExecutionOutputLink
    """
    schema = atomspace.add_node(types.GroundedSchemaNode, "py:relate")
    lst = atomspace.add_link(types.ListLink, [atomspace.add_node(types.ConceptNode, relate_argument),
                                              exec_out_sub])
    return atomspace.add_link(types.ExecutionOutputLink, [schema, lst])


def build_intersect(atomspace, arg0, arg1):
    """
    Build ExecutionOutputLink which accepts two atoms
    and calls intersect callback
    :param atomspace: AtomSpace
    :param arg0: Atom
    :param arg1: Atom
    :return: ExecutionOutputLink
    """
    schema =  atomspace.add_node(types.GroundedSchemaNode, "py:intersect")
    lst = atomspace.add_link(types.ListLink, [arg0,
                                              arg1])
    return atomspace.add_link(types.ExecutionOutputLink, [schema, lst])


def build_bind_link(atomspace, eval_link, inheritance):
    """
    Build bind link from ExecutionOutput or Evaluation links and inheritance links

    :param atomspace: AtomSpace
    :param eval_link: ExecutionOutput or EvaluationLink
    :param inheritance: Set[InheritanceLink]
    :return: BindLink
    """
    varlist = []
    for inh in inheritance:
        for atom in inh.get_out():
            if atom.type == types.VariableNode:
                varlist.append(atom)
    variable_list = atomspace.add_link(types.VariableList, varlist)
    list_link = atomspace.add_link(types.ListLink, varlist + [eval_link])
    conj = atomspace.add_link(types.AndLink, [*inheritance, eval_link])
    bind_link = atomspace.add_link(types.BindLink, [variable_list, conj, list_link])
    return bind_link


def build_scene(atomspace, scene):
    """
    Build ExecutionOutputLink which is used to hold intermediate
    attention maps and feature map

    :param atomspace:
    :param scene: VariableNode
        VariableNode("$Scene")
    :return: ExecutionOutputLink
    """
    schema =  atomspace.add_node(types.GroundedSchemaNode, "py:init_scene")
    lst = atomspace.add_link(types.ListLink, [scene])
    return atomspace.add_link(types.ExecutionOutputLink, [schema, lst])


def build_same(atomspace, same_argument, exec_out_sub):
    """
    Build ExecutionOutputLink calling 'same' callback

    :param atomspace:
    :param same_argument:
    :param exec_out_sub:
    :return: ExecutionOutputLink
    """
    schema = atomspace.add_node(types.GroundedSchemaNode, "py:same")
    lst = atomspace.add_link(types.ListLink, [atomspace.add_node(types.ConceptNode, same_argument),
                                              exec_out_sub])
    return atomspace.add_link(types.ExecutionOutputLink, [schema, lst])


def return_prog(atomspace, commands, inheritance_set=None):
    """
    Convert tbd program to atomese

    :param atomspace: Atomspace
    :param commands: list
    :param inheritance_set: set
        Inhertiance links constructed during conversion
    :return: Tuple[ExecutionOutputLink, list, set]
        program in form of execution output link, unprocessed part of the program, set of inheritance links
    """
    current, rest = commands[0], commands[1:]
    if inheritance_set is None:
        inheritance_set = set()
    scene = atomspace.add_node(types.VariableNode, "$Scene")
    if current.startswith('query'):
        query_type = current.split('_')[-1]
        sub_prog, left, inh = return_prog(atomspace, rest)
        inheritance_set |= inh
        var = atomspace.add_node(types.VariableNode, "$X")
        concept = atomspace.add_node(types.ConceptNode, query_type)
        inh_link = atomspace.add_link(types.InheritanceLink, [var, concept])
        inheritance_set.add(inh_link)
        # todo: use build link build_eval_link(atomspace, variable=var, category=concept, eval_link_sub=sub_prog)
        link = build_filter(atomspace, concept, var, exec_out_sub=sub_prog)
        return link, left, inheritance_set
    elif current.startswith('scene'):
        concept = atomspace.add_node(types.ConceptNode, "BoundingBox")
        inh_link = atomspace.add_link(types.InheritanceLink, [scene, concept])
        inheritance_set.add(inh_link)
        result = build_scene(atomspace, scene)
        return result, rest, inheritance_set
    elif current.startswith('filter'):
        filter_type, filter_arg = filter_reg.match(current).groups()
        sub_prog, left, inh = return_prog(atomspace, rest)
        filter_type_atom = atomspace.add_node(types.ConceptNode, filter_type)
        filter_arg_atom = atomspace.add_node(types.ConceptNode, filter_arg)
        exec_link = build_filter(atomspace, filter_type=filter_type_atom,
                                 filter_argument=filter_arg_atom, exec_out_sub=sub_prog)
        inheritance_set |= inh
        return exec_link, left, inheritance_set
    elif current.startswith('relate'):
        relate_arg = relate_reg.match(current).groups()[0]
        sub_prog, left, inh = return_prog(atomspace, rest)
        inheritance_set |= inh
        return build_relate(atomspace, relate_argument=relate_arg,
                            exec_out_sub=sub_prog), left, inheritance_set
    elif current.startswith('same'):
        same_arg = current.split('_')[-1]
        sub_prog, left, inh = return_prog(atomspace, rest)
        inheritance_set |= inh
        return build_same(atomspace, same_argument=same_arg,
                          exec_out_sub=sub_prog), left, inheritance_set
    elif current.startswith('intersect'):
        sub_prog0, left, inh = return_prog(atomspace, rest)
        inheritance_set |= inh
        sub_prog1, right, inh = return_prog(atomspace, left)
        inheritance_set |= inh
        return build_intersect(atomspace, arg0=sub_prog0, arg1=sub_prog1), right, inheritance_set
    elif current == '<START>':
        return return_prog(atomspace, rest)
    elif current == 'unique':
        return return_prog(atomspace, rest)
    else:
        raise NotImplementedError(current)


# CALLBACKS


def init_scene(scene):
    """
    Accept scene atom and generate new atom which holds dummy attention map
    and features from scene
    :param scene: Atom
    :return: Atom
        An atom with features, attention map, and size for features and attention map
    """
    atomspace = scene.atomspace

    data_atom = atomspace.add_node(types.ConceptNode, 'Data-' + str(uuid.uuid4()))
    key_attention, key_scene, key_shape, key_shape_scene = generate_keys(atomspace)

    data_atom.set_value(key_scene, scene.get_value(key_scene))
    data_atom.set_value(key_shape_scene, scene.get_value(key_shape_scene))
    data_atom.set_value(key_attention, FloatValue(list(tbd.ones_var.flatten())))
    data_atom.set_value(key_shape, FloatValue(list(tbd.ones_var.shape)))
    return data_atom


def generate_keys(atomspace):
    """
    Return predicated nodes to be used as keys for attaching values to atoms
    :param atomspace: AtomSpace
    :return: Tuple[Atom]
    """
    key_scene = atomspace.add_node(types.PredicateNode, 'key_scene')
    key_shape_scene = atomspace.add_node(types.PredicateNode, 'key_shape_scene')
    key_attention = atomspace.add_node(types.PredicateNode, 'key_data')
    key_shape_attention = atomspace.add_node(types.PredicateNode, 'key_shape_attention')
    return key_attention, key_scene, key_shape_attention, key_shape_scene


def extract_tensor(atom, key_data, key_shape):
    """
    Convert FloatValue attached to atom to pytorch array
    :param atom:
    :param key_data:
    :param key_shape:
    :return: torch.Tensor
    """
    value = atom.get_value(key_data)
    shape = atom.get_value(key_shape)
    ar_value = numpy.array(value.to_list())
    ar_value = ar_value.reshape([int(x) for x in shape.to_list()])
    return torch.from_numpy(ar_value).double()


def filter_callback(filter_type, filter_type_instance, data_atom):
    """
    Function which applies filtering neural network module

    :param filter_type: Atom
        An atom with name of filter type e.g. color or size
    :param filter_type_instance: Atom
        An atom with name of particular filter instance e.g. red or small
    :param data_atom:
        An atom with attention map and features attached
    :return:
    """
    module_type = 'filter_' + filter_type.name + '[' + filter_type_instance.name + ']'
    run_attention(data_atom, module_type)
    return data_atom


def set_attention_map(data_atom, key_attention, key_shape_attention, attention):
    """
    Attach attention map to atom

    :param data_atom: Atom
    :param key_attention: Atom
        Atom to be used as key for the attention map
    :param key_shape_attention:
        Atom to be used as key for the attention map shape
    :param attention: numpy.array
    :return:
    """
    data_atom.set_value(key_attention, FloatValue(list(attention.flatten())))
    data_atom.set_value(key_shape_attention, FloatValue(list(attention.shape)))


def intersect(arg0, arg1):
    """
    Intersection of attention maps

    :param arg0: Atom
        An atom with attention map and features attached
    :param arg1: Atom
        An atom with attention map and features attached
    :return: Atom
        arg0 with new attention map attached
    """
    atomspace = arg0.atomspace
    key_attention, key_scene, key_shape_attention, key_shape_scene = generate_keys(atomspace)
    feat_attention1 = extract_tensor(arg0, key_attention, key_shape_attention)
    feat_attention2 = extract_tensor(arg1, key_attention, key_shape_attention)
    module = tbd.function_modules['intersect']
    out = module(feat_attention1, feat_attention2)

    set_attention_map(arg0, key_attention, key_shape_attention, out)
    return arg0


def classify(classifier_type, instance, data_atom):
    """
    Same as filter_callback : should be replaced with classifier returning tv
    """
    return filter_callback(classifier_type, instance, data_atom)


def relate(relation, data_atom):
    """
    Function which applies filtering neural network module

    :param relation: Atom
        An atom with name of type of relation e.g. front or left etc.
    :param data_atom: Atom
        An atom with attention map and features attached
    """
    module_type = 'relate[' + relation.name + ']'
    run_attention(data_atom, module_type)
    return data_atom


def same(relation, data_atom):
    """
    Function which applies same neural network module

    :param relation: Atom
        An atom with name of type of 'same' relation e.g. same color or size etc.
    :param data_atom: Atom
        Atom with attention map and features
    :return:
    """
    module_type = 'same_' + relation.name
    run_attention(data_atom, module_type)
    return data_atom


def run_attention(data_atom, module_type):
    """
    Run neural network module which accepts attention map and features
    and produces attention map

    :param data_atom: Atom
        An atom with attached attention map and features
    :param module_type: str
        Module type name: e.g. filter_color[red] or same_size
    :return:
    """
    module = tbd.function_modules[module_type]
    atomspace = data_atom.atomspace
    key_attention, key_scene, key_shape_attention, key_shape_scene = generate_keys(atomspace)
    feat_input = extract_tensor(data_atom, key_scene, key_shape_scene)
    feat_attention = extract_tensor(data_atom, key_attention, key_shape_attention)
    out = module(feat_input.float(), feat_attention.float())
    set_attention_map(data_atom, key_attention, key_shape_attention, out)