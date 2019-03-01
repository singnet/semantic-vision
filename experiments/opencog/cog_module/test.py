import unittest
import torch

from opencog.ure import BackwardChainer
from opencog.atomspace import AtomSpace, types
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.type_constructors import *
from module import CogModule, CogModel, InputModule, InheritanceModule, get_value
from rules import gen_rules

import rules

import __main__
__main__.CogModule = CogModule
__main__.rules = rules

RED = 0
GREEN = 1
BLUE = 2


class GreenPredicate(CogModule):

    def forward(self, x):
        """
        extract green channel and shift it above zero
        """
        mean = x.mean(dim=-1).mean(dim=-1)
        return mean[GREEN]


class TestBasic(unittest.TestCase):

    def setUp(self):
        self.atomspace = AtomSpace()
        self.model = CogModel(self.atomspace)
        initialize_opencog(self.atomspace)

    def test_eval_link(self):
        apple = ConceptNode('apple')
        colors = torch.rand(3, 4, 4)
        inp = InputModule(apple, colors)
        green = GreenPredicate(ConceptNode('green'))
        query = green.evaluate(inp.execute())
        result = self.model.evaluate_atom(query)
        expected = colors.mean(dim=-1).mean(dim=-1)
        delta = 0.00000001
        self.assertTrue(abs(expected[GREEN] - result.mean) < delta)

    def test_rule_engine(self):
        rbs = ConceptNode("pln")
        gen_rules(rbs)
        apple = ConceptNode('apple')
        colors = torch.rand(3, 4, 4)
        colors[GREEN] = 0.8
        colors = colors
        inp = InputModule(apple, colors)
        # red <- color
        # green <- color
        green = GreenPredicate(ConceptNode('green'))
        inh_red = InheritanceLink(ConceptNode("red"), ConceptNode("color"))
        inh_red.tv = TruthValue(0.8, 0.99)
        inh_red = InheritanceModule(inh_red, torch.tensor(0.9))
        inh_green = InheritanceLink(ConceptNode("green"), ConceptNode("color"))
        inh_green.tv = TruthValue(0.8, 0.99)
        inh_green = InheritanceModule(inh_green, torch.tensor(0.6))
        # And(Evaluation(GreenPredicate, apple), Inheritance(green, color))
        conj = AndLink(green.evaluate(inp.execute()), inh_green.execute())

        bc = BackwardChainer(self.atomspace, rbs, conj)
        bc.do_chain()
        result = get_value(bc.get_results().out[0])
        self.assertEqual(torch.min(colors[GREEN].mean(), inh_green.tv), result)


    def tearDown(self):
        self.atomspace = None
        self.model = None
        finalize_opencog()

