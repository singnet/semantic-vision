from opencog.atomspace import AtomSpace, types, PtrValue, valueToPtrValue
from opencog.type_constructors import *
import torch

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


