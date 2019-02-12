import torch
import uuid
from torch.distributions import normal

# Experimental design of cog.Module API for running opencog reasoning from pytorch nn.Module extension

from opencog.atomspace import AtomSpace, types, PtrValue
from opencog.type_constructors import *


# this is a very specialized version; not for general use
def get_cached_value(atom):
    # can be generalized, e.g. ConceptNodes can be converted to their string names,
    # so string can be an argument to forward, while ConceptNode can be an argument to execute
    key = atom.atomspace.add_node(types.PredicateNode, "cogNet")
    value = atom.get_value(key)
    if value is None:
        raise RuntimeError("atom {0} has no value for {1}".format(str(atom), str(key)))
    result = value.value()
    return result

def set_value(atom, value):
    key = atom.atomspace.add_node(types.PredicateNode, "cogNet")
    atom.set_value(key, PtrValue(value))


def unpack_args(*atoms):
    return (get_cached_value(atom) for atom in atoms)


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
        set_value(atom, self)

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

    def call_forward(self, args):
        #print("Args: ", args)
        #todo: check if ListLink
        args = args.out
        result = self.forward(*unpack_args(*args))
            #*(cogm.cached_result for cogm in ...)
        if args:
            atomspace = args[0].atomspace
        else:
            atomspace = self.atom.atomspace
        id = uuid.uuid4()
        res_atom = atomspace.add_node(types.ConceptNode, 'tmp-result-' + str(id))
        set_value(res_atom, result)
        return res_atom

    def call_forward_tv(self, args):
        #print("Args in evaluation link: ", args)
        res_atom = self.call_forward(args)
        v = torch.mean(self.cached_result)
        res_atom.truth_value(v, 1.0) #todo???
        return TruthValue(v)

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

