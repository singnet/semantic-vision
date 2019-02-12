# Experimental design of cog.Module API for running opencog reasoning from pytorch nn.Module extension

from opencog.atomspace import AtomSpace, types, PtrValue
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.type_constructors import *
from opencog.bindlink import execute_atom, satisfaction_link, bindlink

import torch
from torch.distributions import normal
from module import CogModule, execute, get_cached_value, evaluate, InputModule



# example of pytorch modules

class AttentionModule(CogModule):
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
prog1 = net1.execute(inp.execute())
print(prog1)
print(get_cached_value(execute_atom(atomspace, prog1)))

prog2 = net2.execute(inp.execute())
print(get_cached_value(execute_atom(atomspace, prog2)))

bl = BindLink(
    TypedVariableLink(VariableNode("$X"), TypeNode("ConceptNode")),
    #VariableNode("$X"),
    AndLink(
        InheritanceLink(VariableNode("$X"), ConceptNode("color")),
        evaluate(VariableNode("$X"), inp.execute()) #inp.execution() == execute(ConceptNode("image"))
    ),
    VariableNode("$X")
)
print(bindlink(atomspace, bl))

# ----------------

# wrapping torch.tensor tv to make it trainable
# class TensorTVModule(cogModule):
class InheritanceModule(CogModule):
    def __init__(self, atom, init_tv):
        super().__init__(atom)
        self.tv = init_tv
    def forward(self):
        return self.tv

class AndModule(CogModule):
    def forward(self, a, b):
        return a * b # torch.min(a, b)

# inheritance links are concrete links... we can bind specific objects to them
h = InheritanceLink(ConceptNode("red"), ConceptNode("color"))
InheritanceModule(h, torch.tensor([0.95]))
h = InheritanceLink(ConceptNode("green"), ConceptNode("color"))
InheritanceModule(h, torch.tensor([0.93]))
# doesn't work with Evaluate
print("Tensor truth value: ", get_cached_value(execute_atom(atomspace,
    execute(InheritanceLink(ConceptNode("green"), ConceptNode("color"))))))

# AndLinks are created dynamically, in ad hoc fashion for queries, etc.
# it's cumbersome and unnecesary to bind them to individual objects
# h = AndLink(?, ?)
and_net = AndModule(ConceptNode("AndLink")) # GroundedObjectNode would be suitable here

bl = BindLink(
    VariableNode("$X"),
    #TypedVariableLink(VariableNode("$X"), TypeNode("ConceptNode")),
    and_net.evaluate(
        execute(InheritanceLink(VariableNode("$X"), ConceptNode("color"))),
        execute(VariableNode("$X"), inp.execute())
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
        and_net.evaluate(
            execute(InheritanceLink(VariableNode("$X"), ConceptNode("color"))),
            execute(VariableNode("$X"), inp.execute())
        )
    ),
    # another bad thing is that we need to execute it once again; caching might help though
    and_net.evaluate(
            execute(InheritanceLink(VariableNode("$X"), ConceptNode("color"))),
            execute(VariableNode("$X"), inp.execute()))
)
print("bl1")
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
        GroundedSchemaNode("py:CogModule.newLink"),
        #ListLink(QuoteLink(AndLink(VariableNode("$X"), VariableNode("$Y"))))
        #AndLink(VariableNode("$X"), VariableNode("$Y"))
        ListLink(VariableNode("$Y"))
    )
)

print("bl2")
print(bl)
bindlink(atomspace, bl)
