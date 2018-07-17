import torch
import torch.nn as nn
import torch.nn.functional as F

class NetsVocab(nn.Module):
    
    def __init__(self, device):
        super(NetsVocab, self).__init__()
        
        self.device = device
        self.models = nn.ModuleList()
        self.modelIndexByWord = {}

    @classmethod
    def fromWordsVocabulary(cls, vocabulary, featureVectorSize, device):
        netsVocab = cls(device)
        
        netsVocab.vocabulary = vocabulary
        netsVocab.featureVectorSize = featureVectorSize
        netsVocab.initializeModels()
        
        return netsVocab
    
    @classmethod
    def fromStateDict(cls, device, stateDict):
        netsVocab = cls(device)
        
        netsVocab.vocabulary = stateDict['vocabulary']
        netsVocab.featureVectorSize = stateDict['featureVectorSize']
        netsVocab.initializeModels()
        
        netsVocab.load_state_dict(stateDict['pytorch_state_dict'])
        
        return netsVocab
    
    def state_dict(self):
        return {
            'version' : 1,
            'vocabulary': self.vocabulary,
            'featureVectorSize': self.featureVectorSize,
            'pytorch_state_dict': super().state_dict()
            }
    
    def initializeModels(self):
        modelIndex = 0
        for word in self.vocabulary:
            self.models.append(nn.Sequential(
                nn.Linear(self.featureVectorSize, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                # TODO: why it is not included into model but 
                # applied in feed_forward()
                # nn.Sigmoid()
                ).to(self.device))
            self.modelIndexByWord[word] = modelIndex
            modelIndex += 1
    
    def getModelByWord(self, word):
        if word in self.modelIndexByWord:
            return self.models[self.modelIndexByWord[word]]
        else:
            return None
    
    def getModelsByWords(self, words):
        models = []
        for word in words:
            model = self.getModelByWord(word)
            if model is None:
                continue
            models.append()
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
