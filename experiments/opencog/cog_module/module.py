from contextlib import contextmanager
import torch
import uuid
import weakref
from torch.distributions import normal

# Experimental design of cog.Module API for running opencog reasoning from pytorch nn.Module extension

from opencog.atomspace import AtomSpace, types, PtrValue
from opencog.type_constructors import *
from opencog.atomspace import create_child_atomspace
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.bindlink import execute_atom, evaluate_atom


# this is a very specialized version; not for general use
def get_value(atom):
    # can be generalized, e.g. ConceptNodes can be converted to their string names,
    # so string can be an argument to forward, while ConceptNode can be an argument to execute
    key = atom.atomspace.add_node(types.PredicateNode, "cogNet")
    value = atom.get_value(key)
    if value is None:
        raise RuntimeError("atom {0} has no value for {1}".format(str(atom), str(key)))
    result = value.value()
    if isinstance(result, CogModule):
        return result.forward()
    return result


def set_value(atom, value):
    key = atom.atomspace.add_node(types.PredicateNode, "cogNet")
    atom.set_value(key, PtrValue(value))


def unpack_args(*atoms):
    return (get_value(atom) for atom in atoms)


# todo: new nodes probably should be created in temporary atomspace
def evaluate(atom, *args):
    return EvaluationLink(
        GroundedPredicateNode("py:CogModule.callMethod"),
        ListLink(atom,
                 ConceptNode("call_forward_tv"),
                 ListLink(*args)))


def execute(atom, *args):
    return ExecutionOutputLink(
        GroundedSchemaNode("py:CogModule.callMethod"),
        ListLink(atom,
                 ConceptNode("call_forward"),
                 ListLink(*args)))


# todo: separate cog module from its usage + static methods should be moved to module (e.g. cog.Execute)
class CogModule(torch.nn.Module):
    def __init__(self, atom): #todo: atom is optional? if not given, generate by address? string name for concept instead?
        super().__init__()
        self.atom = atom
        set_value(atom, weakref.proxy(self))
        self._cache = dict()

    @staticmethod
    def callMethod(atom, methodname, args):
        key = PredicateNode("cogNet")
        value = atom.get_value(key)
        if value is None:
            raise RuntimeError("atom {0} has no value for {1}".format(str(atom), str(key)))
        obj = value.value()
        return getattr(obj, methodname.name)(args)

    def execute(self, *args):
        return execute(self.atom, *args)

    def evaluate(self, *args):
        return evaluate(self.atom, *args)

    def get_atomspace(self, args):
        if args:
            return args[0].atomspace
        return self.atom.atomspace

    def call_forward(self, args):
        #print("Args: ", args)
        #todo: check if ListLink
        args = args.out
        if tuple(args) in self._cache:
            return self._cache[tuple(args)]
        result = self.forward(*unpack_args(*args))
        atomspace = self.get_atomspace(args)
        # todo: check new atom is not in atomspace
        # use atomspace as cache
        res_atom = ExecutionOutputLink(
                         GroundedSchemaNode("py:CogModule.callMethod"),
                         ListLink(self.atom,
                                  ConceptNode("call_forward"),
                                  ListLink(*args)))
        set_value(res_atom, result)
        self._cache[tuple(args)] = res_atom
        return res_atom

    def call_forward_tv(self, args):
        args = args.out
        atomspace = self.get_atomspace(args)
        tv_tensor = self.forward(*unpack_args(*args))
        v = torch.mean(tv_tensor)
        ev_link = EvaluationLink(
            GroundedPredicateNode("py:CogModule.callMethod"),
            ListLink(self.atom,
                     ConceptNode("call_forward_tv"),
                     ListLink(*args)))
        set_value(ev_link, tv_tensor)
        ev_link.truth_value(v, 1.0) #todo: confidence
        return TruthValue(v)

    def clear_cache(self):
        self._cache = dict()

    @staticmethod
    def newLink(atom):
        print("HERE: ", atom)
        return atom


class InputModule(CogModule):
    def __init__(self, atom, im):
        super().__init__(atom)
        self.im = im

    def forward(self):
        return self.im

    def call_forward(self, args):
        return self.atom


class InheritanceModule(CogModule):
    def __init__(self, atom, init_tv):
        super().__init__(atom)
        self.tv = init_tv

    def forward(self):
        return self.tv

    def execute(self):
        return self.atom

    def update_tv(self):
        self.atom.tv = TruthValue(torch.mean(self.tv), 1.0)


@contextmanager
def tmp_atomspace(atomspace):
    parent_atomspace = atomspace
    atomspace = create_child_atomspace(parent_atomspace)
    initialize_opencog(atomspace)
    try:
        yield atomspace
    finally:
        atomspace.clear()
        finalize_opencog()
        initialize_opencog(parent_atomspace)


class CogModel(torch.nn.Module):
    def __init__(self, atomspace=None):
        super().__init__()
        if atomspace is None:
            atomspace = AtomSpace()
        self.atomspace = atomspace
        self.__modules = set()

    def __setattr__(self, name, value):
        # todo: store in map[atom] -> module
        if isinstance(value, CogModule):
            self.__modules.add(value)
        super().__setattr__(name, value)

    def execute_atom(self, atom, atomspace=None):
        if atomspace is None:
            atomspace = create_child_atomspace(self.atomspace)
        result = execute_atom(atomspace, atom)
        value = get_value(result)
        self.clear_cache()
        atomspace.clear()
        # todo: use ValueOfLink to get tensor value
        return value

    def evaluate_atom(self, atom, atomspace=None):
        if atomspace is None:
            atomspace = create_child_atomspace(self.atomspace)
        result = evaluate_atom(atomspace, atom)
        # todo: use ValueOfLink to get tensor value
        return result

    def clear_cache(self, atom=None):
        # todo: accept atom,
        # clear only modules affected during execution
        for module in self.__modules:
            module.clear_cache()

    def update_tv(self):
        for module in self.__modules:
            if isinstance(module, InheritanceModule):
                module.update_tv()

