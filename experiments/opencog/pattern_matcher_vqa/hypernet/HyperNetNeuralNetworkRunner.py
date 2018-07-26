import logging
import torch

from .dataset import Dictionary
from .model import build_baseline_model
from pycparser.ply.yacc import pickle_protocol

#pathToDictionary = '/mnt/fileserver/shared/datasets/at-on-at-data/dictionary.pkl'
#pathToGlove = '/mnt/fileserver/shared/datasets/at-on-at-data/glove6b_init_300d.npy'
#pathToModel = '/mnt/fileserver/users/daddywesker/work/TaigaExperiments/32/03_experiment_with_more_layers/01_plus1_prob_layer/model_01_max_score_val.pth.tar'

class HyperNetNeuralNetworkRunner():
    
    def __init__(self, pathToDictionary, pathToGlove, pathToModel):
        self.logger = logging.getLogger('HyperNetNeuralNetworkRunner')
        self.dictionary = Dictionary.load_from_file(pathToDictionary)
        self.model = self.loadModel(pathToGlove, pathToModel)

    def loadModel(self, pathToGlove, pathToModel):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = build_baseline_model(19901, [300, 1280], [2048, 1280], 
              [1280, 1280, 500, 100, 1], [2, 2, 5], [0.5, 'weight', 'ReLU'])
        model = model.to(device)
        model.w_embed.init_embedding(pathToGlove)
        model = torch.nn.DataParallel(model).to(device)
        checkpoint = torch.load(pathToModel, map_location=device.type)
        model.load_state_dict(checkpoint['state_dict'])
        
        return model

    def runNeuralNetwork(self, features, word):
        return model(torch.Tensor(features))