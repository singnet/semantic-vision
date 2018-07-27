import logging
import torch
from torch.autograd import Variable

from hypernetimpl.dataset import Dictionary
from hypernetimpl.model import build_baseline_model

#pathToDictionary = '/mnt/fileserver/shared/datasets/at-on-at-data/dictionary.pkl'
#pathToGlove = '/mnt/fileserver/shared/datasets/at-on-at-data/glove6b_init_300d.npy'
#pathToModel = '/mnt/fileserver/users/daddywesker/work/TaigaExperiments/32/03_experiment_with_more_layers/01_plus1_prob_layer/model_01_max_score_val.pth.tar'

class HyperNetNeuralNetworkRunner():
    
    def __init__(self, pathToDictionary, pathToGlove, pathToModel):
        self.logger = logging.getLogger('HyperNetNeuralNetworkRunner')
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dictionary = Dictionary.load_from_file(pathToDictionary)
        self.model = self.loadModel(pathToGlove, pathToModel)

    def loadModel(self, pathToGlove, pathToModel):
        model = build_baseline_model(19901, [300, 1280], [2048, 1280], 
              [1280, 1280, 500, 100, 1], [2, 2, 5], [0.5, 'weight', 'ReLU'])
        model = model.to(self.device)
        model.w_embed.init_embedding(pathToGlove)
        model = torch.nn.DataParallel(model).to(self.device)
        checkpoint = torch.load(pathToModel, map_location=self.device.type)
        #model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint)
        model.train(False)
        
        return model

    def getTensorByWord(self, word):
        try:
            index = self.dictionary.idx2word.index(word.lower())
            tensor = torch.LongTensor([index])
            return Variable(tensor).to(self.device)
        except ValueError:
            return None

    def runNeuralNetwork(self, features, word):
        wordTensor = self.getTensorByWord(word)
        if wordTensor is None:
            self.logger.debug("Unknown word: %s", word)
            return torch.zeros(1);
        featuresTensor = torch.Tensor(features)
        return self.model(wordTensor, featuresTensor)