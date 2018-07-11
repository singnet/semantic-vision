import torch
import torch.nn as nn
import torch.nn.functional as F

class NetsVocab(nn.Module):
    
    def __init__(self):
        super(NetsVocab, self).__init__()
        
        self.models = nn.ModuleList()
        self.modelIndexByWord = {}
    
    def __init__(self, vocabulary, featureVectorSize, device):
        self.__init__()
        self.device = device
        
        modelIndex = 0
        for word in vocabulary:
            self.models.append(nn.Sequential(
                nn.Linear(featureVectorSize, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                # TODO: why it is not included into model but 
                # applied in feed_forward()
                # nn.Sigmoid()
                ).to(self.device))
            self.modelIndexByWord[word] = modelIndex
            ++modelIndex
    
    def getModelByWord(self, word):
        return self.models[self.modelIndexByWord[word]]
    
    def getModelsByWords(self, words):
        models = []
        for word in words:
            try:
                models.append(self.getModelByWord(word))
            except ValueError:
                continue
        return models
    
    def feed_forward(self, nBBox, x, words):
        output = torch.ones(size=(nBBox,1)).to(self.device)
        for model in self.getModelsByWords(words):
            logits = model(x)
            # logits = model(x).view(-1)
            predict = F.sigmoid(logits)
            output = torch.mul(output, predict)
        return output

    def getParams(self, words):
        params=[]
        for model in self.getModelsByWords(words):
            params.append({'params': model.parameters()})
        return params
