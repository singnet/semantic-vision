import torch
from torch.distributions import normal

# Experimental design of cog.Module API for running opencog reasoning from pytorch nn.Module extension

from opencog.atomspace import AtomSpace, types, PtrValue, valueToPtrValue
from opencog.type_constructors import *


# this is a very specialized version; not for general use
def get_cached_value(atom):
    # can be generalized, e.g. ConceptNodes can be converted to their string names,
    # so string can be an argument to forward, while ConceptNode can be an argument to execute
    value = atom.get_value(atom.atomspace.add_node(types.PredicateNode, "cogNet"))
    print(value)
    result = valueToPtrValue(value).value().cached_result
    return result


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
        atom.set_value(PredicateNode("cogNet"), PtrValue(self))

    @staticmethod
    def callMethod(atom, methodname, args):
        key = PredicateNode("cogNet")
        value = atom.get_value(key)
        if value is None:
            raise RuntimeError("atom {0} has no value for {1}".format(str(atom), str(key)))
        obj = valueToPtrValue(value).value()
        return getattr(obj, methodname.name)(args)

    def execute(self, *args):
        return execute(self.atom, *args)

    def evaluate(self, *args):
        return evaluate(self.atom, *args)

    def call_forward(self, args):
        #print("Args: ", args)
        #todo: check if ListLink
        args = args.out
        self.cached_result = self.forward(*unpack_args(*args))
            #*(cogm.cached_result for cogm in ...)
        return self.atom

    def call_forward_tv(self, args):
        self.call_forward(args)
        v = torch.mean(self.cached_result)
        self.atom.truth_value(v, 1.0) #todo???
        return TruthValue(v)

    @staticmethod
    def newLink(atom):
        print("HERE: ", atom)
        return atom


class InputModule(CogModule):
    def __init__(self, atom, im):
        super().__init__(atom)
        self.im = im
    # def set_input(self, im) -- can be a method in cogModule
    # once called, id of the current input is increase to re-call forward() from execute(),
    # otherwise cached result can be returned... id trees can be automatically constructed by execute()...
    def forward(self):
        return self.im
