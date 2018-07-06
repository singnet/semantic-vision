import logging
import re

import torch.nn as nn

from opencog.atomspace import AtomSpace
from opencog.type_constructors import *
from opencog.bindlink import evaluate_atom, execute_atom

class ModelVocabulary(nn.Module):
    
    def __init__(self, words):
        super().__init__()
        
        self.models = nn.ModuleList()
        self.modelIndexByWord = {}
        
        modelIndex = 0
        for word in words:
            self.models.append(nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                ).to(device))
            self.modelIndexByWord[word] = modelIndex
            ++modelIndex
    
    def getModelByWord(self, word):
        return self.models[self.modelIndexByWord[word]]
#
#     def getParams(self, idx):
#         # TODO: fix
#         params=[]
#         for i in idx:
#             params.append({'params': self.models[i].parameters()})
#         return params

def loadFeatures():
    return []

def buildModelFromAtomeese(modelCombination):
    global models
    combined = torch.ones(size=(nBBox,1)).to(device)
    andLink = modelCombination
    for conceptNode in andLink.get_out():
        # TODO: can be faster if regexp is compiled
        match = re.search('^(.+)(?=.nn$)', conceptNode.get_name())
        word = match.group(0)
        model = models.getModelByWord(word)
        combined = torch.max(torch.mul(nn.Sequential(model, nn.Sigmoid)), 0)
    return combined

def runNeuralNetwork(modelCombination, featuresContainer):
    finalModel = buildModelFromAtomeese(modelCombination)
    features = featuresContainer.get_value(
        PredicateNode('neuralNetworkFeatures'))
    value = finalModel(features)
    return TruthValue(value, 1.0)

def trainNeuralNetwork(modelCombination, predictedValue, expectedValue):
    pass

def initLogger():
    global log
    log = logging.getLogger('training_steps')
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler())

def initAtomspace():
    global atomspace
    atomspace = AtomSpace()
    set_type_ctor_atomspace(atomspace)

initLogger()
initAtomspace()

models = ModelVocabulary(['hare', 'grey'])

boundingBox = ConceptNode("someBoundingBox")
boundingBox.set_value(PredicateNode("neuralNetworkFeatures"), 
                      FloatValue(loadFeatures()))

modelCombination = AndLink(ConceptNode('hare.nn'), ConceptNode('grey.nn'))
log.info("modelCombination: %s", modelCombination)

expectedValue = TruthValue(1.0, 1.0)
log.info('expectedValue: %s', expectedValue)

predictedValue = evaluate_atom(atomspace, EvaluationLink(
    GroundedPredicateNode('py:runNeuralNetwork'), 
    ListLink(modelCombination, boundingBox)
    ))
log.info('predictedValue: %s', predictedValue)

# TODO: there is no Atomeese construction to call Python procedure 
# but don't expect any result, i.e. call ```void foo()```
execute_atom(atomspace, EvaluationLink(
    GroundedPredicateNode('py:trainNeuralNetwork'), 
    ListLink(modelCombination, predictedValue, expectedValue)
    ))