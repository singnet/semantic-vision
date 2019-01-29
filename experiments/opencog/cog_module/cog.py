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
    def newLink(atom):
        print("HERE: ", atom)
        return atom

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

    def Eval(self, *args):
        return cogModule.Evaluate(self.atom, *args)

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
    TypedVariableLink(VariableNode("$X"), TypeNode("ConceptNode")),
    #VariableNode("$X"),
    AndLink(
        InheritanceLink(VariableNode("$X"), ConceptNode("color")),
        cogModule.Evaluate(VariableNode("$X"), inp.Exec()) #inp.Exec() == cogModule.Execute(ConceptNode("image"))
    ),
    VariableNode("$X")
)
print(bindlink(atomspace, bl))

# ----------------

# wrapping torch.tensor tv to make it trainable
# class TensorTVModule(cogModule):
class InheritanceModule(cogModule):
    def __init__(self, atom, init_tv):
        super().__init__(atom)
        self.tv = init_tv
    def forward(self):
        return self.tv

class AndModule(cogModule):
    def forward(self, a, b):
        return a * b # torch.min(a, b)

# inheritance links are concrete links... we can bind specific objects to them
h = InheritanceLink(ConceptNode("red"), ConceptNode("color"))
InheritanceModule(h, torch.tensor([0.95]))
h = InheritanceLink(ConceptNode("green"), ConceptNode("color"))
InheritanceModule(h, torch.tensor([0.93]))
# doesn't work with Evaluate
print("Tensor truth value: ", get_cached_value(execute_atom(atomspace,
    cogModule.Execute(InheritanceLink(ConceptNode("green"), ConceptNode("color"))))))

# AndLinks are created dynamically, in ad hoc fashion for queries, etc.
# it's cumbersome and unnecesary to bind them to individual objects
# h = AndLink(?, ?)
and_net = AndModule(ConceptNode("AndLink")) # GroundedObjectNode would be suitable here

bl = BindLink(
    VariableNode("$X"),
    #TypedVariableLink(VariableNode("$X"), TypeNode("ConceptNode")),
    and_net.Eval(
        cogModule.Execute(InheritanceLink(VariableNode("$X"), ConceptNode("color"))),
        cogModule.Execute(VariableNode("$X"), inp.Exec())
    ),
    VariableNode("$X")
)
# doesn't work, because Pattern Matcher doesn't see cogModule.Evaluate(InheritanceLink...) and cogModule.Evaluate(VariableNode...) as clauses,
# so no restrictions on $X
#print(bindlink(atomspace, bl))


bl = BindLink(
    VariableNode("$X"),
    AndLink(
        InheritanceLink(VariableNode("$X"), ConceptNode("color")),
        and_net.Eval(
            cogModule.Execute(InheritanceLink(VariableNode("$X"), ConceptNode("color"))),
            cogModule.Execute(VariableNode("$X"), inp.Exec())
        )
    ),
    # another bad thing is that we need to execute it once again; caching might help though
    and_net.Eval(
            cogModule.Execute(InheritanceLink(VariableNode("$X"), ConceptNode("color"))),
            cogModule.Execute(VariableNode("$X"), inp.Exec()))
)
print(bl)
print(bindlink(atomspace, bl))
# somehow works, but:
# - Execute should be used instead of Evaluate in internal calls - that's ok, and
# - looks somewhat artificial
# - and_net.Eval result is lost... (and it is assigned to Node, so we get recover only one value for all results...)
# ==> URE is necessary

AndLink(
    InheritanceLink(ConceptNode("a"), ConceptNode("b")),
    EvaluationLink(PredicateNode("c"), ConceptNode("d")))

bl = BindLink(
    VariableList(
        TypedVariableLink(VariableNode("$X"), TypeNode("InheritanceLink")),
        TypedVariableLink(VariableNode("$Y"), TypeNode("EvaluationLink"))),
        #TypedVariableLink(VariableNode("$X"), TypeChoice(TypeNode("ExecutionOutputLink"), TypeNode("InheritanceLink"))),
        #TypedVariableLink(VariableNode("$Y"), TypeChoice(TypeNode("ExecutionOutputLink"), TypeNode("InheritanceLink")))),
    AndLink(
        AndLink(
            VariableNode("$X"),
            VariableNode("$Y")
        )
    ),
    ExecutionOutputLink(
        GroundedSchemaNode("py:cogModule.newLink"),
        #ListLink(QuoteLink(AndLink(VariableNode("$X"), VariableNode("$Y"))))
        #AndLink(VariableNode("$X"), VariableNode("$Y"))
        ListLink(VariableNode("$Y"))
    )
)

bindlink(atomspace, bl)
