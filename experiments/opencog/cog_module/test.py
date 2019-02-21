import unittest
import torch

from opencog.atomspace import AtomSpace, types
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.type_constructors import *
from module import CogModule, CogModel, InputModule
from module import CogModule


import __main__
__main__.CogModule = CogModule


RED = 0
GREEN = 1
BLUE = 2


class GreenPredicate(CogModule):

    def forward(self, x):
        """
        extract green channel and shift it above zero
        """
        mean = x.mean(dim=-1).mean(dim=-1)
        return mean[GREEN] + x.max()

class TestBasic(unittest.TestCase):

    def setUp(self):
        self.atomspace = AtomSpace()
        self.model = CogModel(self.atomspace)
        initialize_opencog(self.atomspace)

    def test_eval_link(self):
        apple = ConceptNode('apple')
        colors = torch.rand(3, 4, 4) - 0.5
        inp = InputModule(apple, colors)
        green = GreenPredicate(ConceptNode('green'))
        query = green.evaluate(inp.execute())
        result = self.model.evaluate_atom(query)
        expected = colors.mean(dim=-1).mean(dim=-1) + colors.max()
        delta = 0.00000001
        self.assertTrue(expected[GREEN] - result.mean < delta)

    def tearDown(self):
        self.atomspace = None
        self.model = None
        finalize_opencog()

