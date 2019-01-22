# Experimental design of cog.Module API for running opencog reasoning from pytorch nn.Module extension

from opencog.atomspace import AtomSpace, types, PtrValue, valueToPtrValue
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.type_constructors import *
from opencog.bindlink import execute_atom, satisfaction_link, bindlink

import torch
from torch.distributions import normal

# this is a very specialized version; not for general use
def get_cached_value(atom):
    # can be generalized, e.g. ConceptNodes can be converted to their string names,
    # so string can be an argument to forward, while ConceptNode can be an argument to execute
    value = atom.get_value(PredicateNode("cogNet"))
    result = valueToPtrValue(value).value().cached_result
    return result


def unpack_args(*atoms):
    return (get_cached_value(atom) for atom in atoms)

# todo: separate cog module from its usage + static methods should be moved to module (e.g. cog.Execute)

class cogModule(torch.nn.Module):
    def __init__(self, atom): #todo: atom is optional? if not given, generate by address? string name for concept instead?
        super().__init__()
        self.atom = atom
        atom.set_value(PredicateNode("cogNet"), PtrValue(self))

    @staticmethod
    def callMethod(atom, methodname, args):
        obj = valueToPtrValue(atom.get_value(PredicateNode("cogNet"))).value()
        return getattr(obj, methodname.name)(args)

    @staticmethod
    def Execute(atom, *args):
        return ExecutionOutputLink(
            GroundedSchemaNode("py:cogModule.callMethod"),
            ListLink(atom,
                     ConceptNode("execute"),
                     ListLink(*args)))
    
    @staticmethod
    def Evaluate(atom, *args):
        return EvaluationLink(
            GroundedPredicateNode("py:cogModule.callMethod"),
            ListLink(atom,
                     ConceptNode("evaluate"),
                     ListLink(*args)))


    #todo: same names with static Execute ?
    def Exec(self, *args):
        return cogModule.Execute(self.atom, *args)

    def execute(self, args):
        #print("Args: ", args)
        #todo: check if ListLink
        args = args.out
        if(len(args) > 0):
            self.cached_result = self.forward(*unpack_args(*args))
            #*(cogm.cached_result for cogm in ...)
        else:
            self.cached_result = self.forward()
        return self.atom
    
    def evaluate(self, args):
        #print("Args: ", args)
        self.execute(args)
        v = torch.mean(self.cached_result)
        self.atom.truth_value(v, 1.0) #todo???
        return TruthValue(v)

class InputModule(cogModule):
    def __init__(self, atom, im):
        super().__init__(atom)
        self.im = im
    # def set_input(self, im) -- can be a method in cogModule
    # once called, id of the current input is increase to re-call forward() from execute(),
    # otherwise cached result can be returned... id trees can be automatically constructed by execute()...
    def forward(self):
        return self.im

class AttentionModule(cogModule):
    def __init__(self, atom):
        super().__init__(atom)
        self.x = normal.Normal(0.0, 1.0).sample()
    def forward(self, xs):
        #print("xs=", xs)
        return xs + self.x

atomspace = AtomSpace()
initialize_opencog(atomspace)

inp = InputModule(ConceptNode("image"), torch.tensor([1.]))
InheritanceLink(ConceptNode("red"), ConceptNode("color"))
InheritanceLink(ConceptNode("green"), ConceptNode("color"))
net1 = AttentionModule(ConceptNode("red"))
net2 = AttentionModule(ConceptNode("green"))


# direct execution proceeds as usual
print(net1(inp()))

# execution from Atomese
prog1 = net1.Exec(inp.Exec())
print(prog1)
print(get_cached_value(execute_atom(atomspace, prog1)))

prog2 = net2.Exec(inp.Exec())
print(get_cached_value(execute_atom(atomspace, prog2)))

bl = BindLink(
    #TypedVariableNode(VariableNode("$X"), TypeNode("ConceptNode")),
    VariableNode("$X"),
    AndLink(
        InheritanceLink(VariableNode("$X"), ConceptNode("color")),
        cogModule.Evaluate(VariableNode("$X"), inp.Exec()) #inp.Exec() == cogModule.Execute(ConceptNode("image"))
    ),
    VariableNode("$X")
)
bindlink(atomspace, bl)
