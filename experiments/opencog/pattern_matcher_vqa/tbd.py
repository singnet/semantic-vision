import re

from opencog.atomspace import AtomSpace, types
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
    schema = atomspace.add_node(types.GroundedSchemaNode, "py:filter")

    lst = atomspace.add_link(types.ListLink, [atomspace.add_node(types.ConceptNode, filter_type),
                                              atomspace.add_node(types.ConceptNode, filter_argument),
                                              exec_out_sub])
    return atomspace.add_link(types.ExecutionOutputLink, [schema, lst])


def build_relate(atomspace, relate_argument, exec_out_sub):
    schema =  atomspace.add_node(types.GroundedSchemaNode, "py:relate")
    lst = atomspace.add_link(types.ListLink, [atomspace.add_node(types.ConceptNode, relate_argument),
                                              exec_out_sub])
    return atomspace.add_link(types.ExecutionOutputLink, [schema, lst])


def build_intersect(atomspace, arg0, arg1):
    schema =  atomspace.add_node(types.GroundedSchemaNode, "py:intersect")
    lst = atomspace.add_link(types.ListLink, [arg0,
                                              arg1])
    return atomspace.add_link(types.ExecutionOutputLink, [schema, lst])


def build_bind_link(atomspace, eval_link, inheritance):
    varlist = []
    for inh in inheritance:
        for atom in inh.get_out():
            if atom.type == types.VariableNode:
                varlist.append(atom)
    variable_list = atomspace.add_link(types.VariableList, varlist)
    conj = atomspace.add_link(types.AndLink, [*inheritance, eval_link])
    bind_link = atomspace.add_link(types.BindLink, [variable_list, conj, variable_list])
    return bind_link


def return_prog(atomspace, commands, inheritance_set=None):
    current, rest = commands[0], commands[1:]
    if inheritance_set is None:
        inheritance_set = set()
    if current.startswith('query'):
        query_type = current.split('_')[-1]
        sub_prog, left, inh = return_prog(atomspace, rest)
        inheritance_set |= inh
        var = atomspace.add_node(types.VariableNode, "$X")
        concept = atomspace.add_node(types.ConceptNode, query_type)
        inh_link = atomspace.add_link(types.InheritanceLink, [var, concept])
        inheritance_set.add(inh_link)
        link = build_eval_link(atomspace, variable=var, eval_link_sub=sub_prog)
        return link, left, inheritance_set
    elif current.startswith('scene'):
        var = atomspace.add_node(types.VariableNode, "$Scene")
        concept = atomspace.add_node(types.ConceptNode, "BoundingBox")
        inh_link = atomspace.add_link(types.InheritanceLink, [var, concept])
        inheritance_set.add(inh_link)
        return var, rest, inheritance_set
    elif current.startswith('filter'):
        filter_type, filter_arg = filter_reg.match(current).groups()
        sub_prog, left, inh = return_prog(atomspace, rest)
        exec_link = build_filter(atomspace, filter_type=filter_type,
                                 filter_argument=filter_arg, exec_out_sub=sub_prog)
        inheritance_set |= inh
        return exec_link, left, inheritance_set
    elif current.startswith('relate'):
        relate_arg = relate_reg.match(current).groups()[0]
        sub_prog, left, inh = return_prog(atomspace, rest)
        inheritance_set |= inh
        return build_relate(atomspace, relate_argument=relate_arg,
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


def init_scene(scene):
    """
    Accept scene atom and generate new atom which holds dummy attention map
    and features from scene
    :param scene: Atom
    :return: Atom
        An atom with features, attention map, and size for features and attention map
    """
    atomspace = scene.atomspace
    data_atom = atomspace.add_node(types.ConceptNode, 'Data')
    key_attention, key_scene, key_shape, key_shape_scene = generate_keys(atomspace)

    data_atom.set_value(key_scene, scene.get_value(key_scene))
    data_atom.set_value(key_shape_scene, scene.get_value(key_shape_scene))
    data_atom.set_value(key_attention, FloatValue(list(tbd.ones_var.flatten())))
    data_atom.set_value(key_shape, FloatValue(list(tbd.ones_var.shape)))
    return data_atom


def generate_keys(atomspace):
    key_scene = atomspace.add_node(types.PredicateNode, 'key_scene')
    key_shape_scene = atomspace.add_node(types.PredicateNode, 'key_shape_scene')
    key_attention = atomspace.add_node(types.PredicateNode, 'key_data')
    key_shape_attention = atomspace.add_node(types.PredicateNode, 'key_shape_attention')
    return key_attention, key_scene, key_shape_attention, key_shape_scene


def extract_tensor(atom, key_data, key_shape):
    value = atom.get_value(key_data)
    shape = atom.get_value(key_shape)
    ar_value = numpy.as_array(value.to_list())
    ar_value = ar_value.reshape([int(x) for x in shape.to_list()])
    return torch.from_numpy(ar_value)


def filter(filter_type, filter_type_instance, data_atom):
    """
    Function which applies filtering neural network module
    """
    module_type = 'filter_' + filter_type + '[' + filter_type_instance + ']'
    module = tbd.function_modules[module_type]

    atomspace = data_atom.atomspace
    key_attention, key_scene, key_shape_attention, key_shape_scene = generate_keys(atomspace)
    feat_input = extract_tensor(data_atom, key_scene, key_shape_scene)
    feat_attention = extract_tensor(data_atom, key_attention, key_shape_attention)
    out = module(feat_input, feat_attention)
    set_attention_map(data_atom, key_attention, key_shape_attention, out)
    return data_atom


def set_attention_map(data_atom, key_attention, key_shape_attention, out):
    data_atom.set_value(key_attention, FloatValue(list(out.flatten())))
    data_atom.set_value(key_shape_attention, FloatValue(list(out.shape)))


def intersect(arg0, arg1):
    """
    Intersection of attention maps
    :param arg0:
    :param arg1:
    :return: Atom
    """
    atomspace = data_atom.atomspace
    key_attention, key_scene, key_shape_attention, key_shape_scene = generate_keys(atomspace)
    feat_attention1 = extract_tensor(arg0, key_attention, key_shape_attention)
    feat_attention2 = extract_tensor(arg1, key_attention, key_shape_attention)
    module = tbd.function_modules['intersect']
    out = module(feat_attention1, feat_attention2)

    set_attention_map(arg0, key_attention, key_shape_attention, out)
    return arg0


def classify(classifier_type, instance, data_atom):
    """
    Same as filter: should be replaced with classifier returning tv
    """
    return filter(classifier_type, instance, data_atom)
