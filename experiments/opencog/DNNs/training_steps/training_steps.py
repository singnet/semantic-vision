import logging
import re
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencog.atomspace import AtomSpace
from opencog.type_constructors import *
from opencog.bindlink import evaluate_atom, execute_atom
from asyncio.events import Handle

class WordModelVocabulary:
    
    def __init__(self, words):
        super().__init__()
        
        self.models = nn.ModuleList()
        self.modelIndexByWord = {}
        
        global featureVectorSize, torchDevice
        modelIndex = 0
        for word in words:
            self.models.append(nn.Sequential(
                nn.Linear(featureVectorSize, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
                ).to(torchDevice))
            self.modelIndexByWord[word] = modelIndex
            ++modelIndex
    
    def getModelByWord(self, word):
        return self.models[self.modelIndexByWord[word]]

class TensorCache:
    
    def __init__(self):
        self.nextHandle = 0
        self.cache = {}
    
    def addTensor(self, tensor):
        handle = self.nextHandle
        self.nextHandle += 1
        self.cache[handle] = tensor
        return handle
    
    def getTensor(self, handle):
        return self.cache[handle]
    
    def clearTensor(self, handle):
        del self.cache[handle]

tensorCache = TensorCache()

def getTensorValue(atom):
    global tensorCache
    tensorHandle = int(atom.get_value(PredicateNode('tensorHandle'))
                       .to_list()[0])
    tensor = tensorCache.getTensor(tensorHandle)
    tensorCache.clearTensor(tensorHandle)
    return tensor

def setTensorValue(atom, tensor):
    global tensorCache
    handle = tensorCache.addTensor(tensor)
    atom.set_value(PredicateNode('tensorHandle'), FloatValue(handle))

def loadFeatures():
    global featureVectorSize
    return list(map(lambda x: random.random(), range(featureVectorSize)))

def applyAtomeeseModel(atomeeseModel, features):
    global wordModels, torchDevice
    tensor = torch.ones(1).to(torchDevice)
    andLink = atomeeseModel
    for conceptNode in andLink.get_out():
        # TODO: can be faster if regexp is compiled
        match = re.search('^(.+)(?=.nn$)', conceptNode.name)
        word = match.group(0)
        model = wordModels.getModelByWord(word)
        tensor = torch.mul(model(features), tensor)
    return tensor

def runNeuralNetwork(atomeeseModel, featuresContainer):
    features = torch.tensor(featuresContainer.get_value(
        PredicateNode('neuralNetworkFeatures')).to_list())
    tensor = applyAtomeeseModel(atomeeseModel, features)
    result = EvaluationLink(PredicateNode('neuralNetworkPrediction'), 
                             ListLink(atomeeseModel, featuresContainer))
    result.tv = TruthValue(tensor.item(), 1.0)
    setTensorValue(result, tensor)
    return result

def createExpectedResult(atomeeseModel, featuresContainer, truthValue):
    result = EvaluationLink(PredicateNode('expectedNeuralNetworkAnswer'), 
                             ListLink(atomeeseModel, featuresContainer))
    result.tv = TruthValue(truthValue, 1.0)
    setTensorValue(result, torch.tensor([ truthValue ]))
    return result

def getAtomeeseModelParameters(atomeeseModel):
    parameters = itertools.chain()
    andLink = atomeeseModel
    for conceptNode in andLink.get_out():
        # TODO: can be faster if regexp is compiled
        match = re.search('^(.+)(?=.nn$)', conceptNode.name)
        word = match.group(0)
        model = wordModels.getModelByWord(word)
        parameters = itertools.chain(parameters, model.parameters())
    return parameters

def trainNeuralNetwork(atomeeseModel, predictedValue, expectedValue):
    # TODO: model can be cached in predictedValue as well
    parameters = getAtomeeseModelParameters(atomeeseModel)
    # TODO: Maxim already implemented dynamic learning rate
    optimizer = torch.optim.SGD(parameters, lr=1e-3, momentum=0)
    optimizer.zero_grad()
    loss = F.binary_cross_entropy(getTensorValue(predictedValue),
                                  getTensorValue(expectedValue))
    
    loss.backward()
    optimizer.step()
    
    return TruthValue(1.0, 1.0)

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

featureVectorSize = 2048
torchDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wordModels = WordModelVocabulary(['hare', 'grey'])

boundingBox = ConceptNode("someBoundingBox")
boundingBox.set_value(PredicateNode("neuralNetworkFeatures"), 
                      FloatValue(loadFeatures()))

atomeeseModel = AndLink(ConceptNode('hare.nn'), ConceptNode('grey.nn'))
log.info("atomeeseModel: %s", atomeeseModel)

expectedValue = createExpectedResult(atomeeseModel, boundingBox, 1.0)
log.info('expectedValue: %s', expectedValue)

predictedValue = execute_atom(atomspace, ExecutionOutputLink(
    GroundedSchemaNode('py:runNeuralNetwork'), 
    ListLink(atomeeseModel, boundingBox)
    ))
log.info('predictedValue: %s', predictedValue)

# TODO: there is no Atomeese construction to call Python procedure 
# but don't expect any result, i.e. call ```void foo()```
evaluate_atom(atomspace, EvaluationLink(
    GroundedPredicateNode('py:trainNeuralNetwork'), 
    ListLink(atomeeseModel, predictedValue, expectedValue)
    ))
log.info('train step finished')