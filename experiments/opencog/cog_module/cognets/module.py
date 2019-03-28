from enum import Enum
from contextlib import contextmanager
import torch
import weakref

# Experimental design of cog.Module API for running opencog reasoning from pytorch nn.Module extension

from opencog.atomspace import types, PtrValue, Atom
from opencog.type_constructors import *
from opencog.atomspace import create_child_atomspace
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.bindlink import execute_atom, evaluate_atom


class EVALMODE(Enum):
    FEATUREMAP = 1
    MEAN = 2
    STV = 3


MEAN = 0
CONFIDENCE = 1


def get_value(atom, tv=False):
    if tv:
        key = atom.atomspace.add_node(types.PredicateNode, "cogNet-tv")
    else:
        key = atom.atomspace.add_node(types.PredicateNode, "cogNet")
    value = atom.get_value(key)
    if value is None:
        if tv:
            default = TTruthValue([atom.tv.mean, atom.tv.confidence])
            set_value(atom, default, tv=tv)
            return default
        return value
    result = value.value()
    if isinstance(result, CogModule):
        return result.forward()
    return result


def set_value(atom, value, tv=False):
    if tv:
        key = atom.atomspace.add_node(types.PredicateNode, "cogNet-tv")
        assert isinstance(value, TTruthValue) or isinstance(value, CogModule)
    else:
        key = atom.atomspace.add_node(types.PredicateNode, "cogNet")
    atom.set_value(key, PtrValue(value))


def set_tv(atom, value):
    return set_value(atom, value, tv=True)


def get_tv(atom):
    return get_value(atom, tv=True)


def unpack_args(*atoms, tv=False):
    """
    Return attached tensor, if tv=True expected tensor is truth value,
    so default value will be created in case of no tensor being attach to an atom
    """
    return [get_value(atom, tv=tv) for atom in atoms]


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
    def __init__(self, atom, ev_mode=EVALMODE.MEAN): #todo: atom is optional? if not given, generate by address? string name for concept instead?
        super().__init__()
        self.atom = atom
        set_value(atom, weakref.proxy(self))
        self.eval_mode = ev_mode

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
        args = args.out
        res_atom = ExecutionOutputLink(
                         GroundedSchemaNode("py:CogModule.callMethod"),
                         ListLink(self.atom,
                                  ConceptNode("call_forward"),
                                  ListLink(*args)))
        cached_value = get_value(res_atom)
        if cached_value is not None:
            return res_atom
        result = self.forward(*unpack_args(*args))
        set_value(res_atom, result)
        return res_atom

    def call_forward_tv(self, args):
        args = args.out
        ev_link = EvaluationLink(
             GroundedPredicateNode("py:CogModule.callMethod"),
             ListLink(self.atom,
                      ConceptNode("call_forward_tv"),
                      ListLink(*args)))
        cached_value = get_value(ev_link)
        if cached_value is not None:
            return ev_link.tv
        out = self.forward(*unpack_args(*args))
        if self.eval_mode == EVALMODE.MEAN:
            assert len(out.shape) == 0
            tv_tensor = TTruthValue([out, torch.tensor(1.0)])
        elif self.eval_mode == EVALMODE.STV:
            assert len(out) == 2
            tv_tensor = TTruthValue(out)
        else:
            raise NotImplementedError("mode not implemented")
        set_tv(ev_link, tv_tensor)
        ev_link.tv = TruthValue(tv_tensor[MEAN], tv_tensor[CONFIDENCE])
        return ev_link.tv


class InputModule(CogModule):
    def __init__(self, atom, im):
        super().__init__(atom)
        self.im = im

    def forward(self):
        return self.im

    def call_forward(self, args):
        return self.atom


class InheritanceModule(CogModule):
    def __init__(self, *args, tv=None):
        assert bool(args)
        if len(args) == 1:
            self.__init_link(args[0], tv)
            return
        assert len(args) == 2
        self.__init_link(InheritanceLink(*args), tv)


    def __init_link(self, inh_link, init_tv=None):
        super().__init__(inh_link)
        if init_tv is None:
            init_tv = TTruthValue([inh_link.tv.mean,
                inh_link.tv.confidence])
        assert len(init_tv) == 2
        self.tv = TTruthValue(init_tv)
        set_tv(inh_link, weakref.proxy(self))
        self.update_tv()

    def forward(self):
        return self.tv

    def evaluate(self):
        return self.atom

    def execute(self):
        return self.atom

    def update_tv(self):
        self.atom.tv = TruthValue(self.tv[MEAN], self.tv[CONFIDENCE])


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
        atomspace.clear()
        # todo: use ValueOfLink to get tensor value
        return value

    def evaluate_atom(self, atom, atomspace=None):
        if atomspace is None:
            atomspace = create_child_atomspace(self.atomspace)
        result = evaluate_atom(atomspace, atom)
        # todo: use ValueOfLink to get tensor value
        return result

    def update_tv(self):
        for module in self.__modules:
            if isinstance(module, InheritanceModule):
                module.update_tv()


class TTruthValue(torch.Tensor):

    @property
    def mean(self):
        return self[MEAN]

    @property
    def confidence(self):
        return self[CONFIDENCE]

