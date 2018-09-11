from abc import ABC


class NeuralNetworkRunner(ABC):
    
    def runNeuralNetwork(self, features, word):
        pass


class AnswerHandler(ABC):
    
    def onNewQuestion(self, record):
        pass
    
    def onAnswer(self, record, answer):
        pass

    def getUnanswered(self):
        return list()


class ChainAnswerHandler(AnswerHandler):
    
    def __init__(self, answerHandlerList):
        self.answerHandlerList = answerHandlerList
        
    def onNewQuestion(self, record):
        self.notifyAll(lambda handler: handler.onNewQuestion(record))

    def onAnswer(self, record, answer):
        self.notifyAll(lambda handler: handler.onAnswer(record, answer))
        
    def notifyAll(self, methodToCall):
        map(methodToCall, answerHandlerList)


class FeatureExtractor(ABC):
    
    def getFeaturesByImageId(self, imageId):
        pass


class NoModelException(RuntimeError):
    pass
