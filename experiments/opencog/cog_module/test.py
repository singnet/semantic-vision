import unittest
import torch

from opencog.ure import BackwardChainer
from opencog.atomspace import AtomSpace, types
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.type_constructors import *
from cognets import CogModule, CogModel, InputModule, InheritanceModule, get_value, TTruthValue
from cognets import get_tv
from pln import initialize_pln

import pln

import __main__
__main__.CogModule = CogModule
__main__.pln = pln

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
        rule_base = initialize_pln()
        apple = ConceptNode('apple')
        colors = torch.rand(3, 4, 4)
        colors[GREEN] = 0.8
        colors = colors
        inp = InputModule(apple, colors)
        # red <- color
        # green <- color
        green = GreenPredicate(ConceptNode('green'))
        inh_red = InheritanceLink(ConceptNode("red"), ConceptNode("color"))
        inh_red = InheritanceModule(inh_red, torch.tensor([0.9, 0.95]))
        inh_green = InheritanceModule(ConceptNode("green"), ConceptNode("color"),
                                      tv=torch.tensor([0.6, .9]))
        # And(Evaluation(GreenPredicate, apple), Inheritance(green, color))
        conj = AndLink(green.evaluate(inp.execute()), inh_green.execute())

        bc = BackwardChainer(self.atomspace, rule_base, conj)
        bc.do_chain()
        result = get_tv(bc.get_results().out[0])
        self.assertEqual(min(colors[GREEN].mean(), inh_green.tv.mean), result.mean)
        self.assertEqual(inh_green.tv.confidence, result.confidence)

    def tearDown(self):
        self.atomspace = None
        self.model = None
        finalize_opencog()

