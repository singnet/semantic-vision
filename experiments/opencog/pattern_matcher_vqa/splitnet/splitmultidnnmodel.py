import json
import torch
import torch.nn as nn
import os
import glob
import re
import sys
import logging

from splitnet.dictionary import Dictionary

from interface import NeuralNetworkRunner, NoModelException
import numpy

sys.path.insert(0, os.path.dirname(__file__) + '/../../DNNs/vqa_multi_dnn')
from netsvocabulary import INetsVocab

logger = logging.getLogger('NetsVocabularyNeuralNetworkRunner')


class SplitNetsVocab(INetsVocab):

    def __init__(self, models_directory):
        super().__init__()
        path = os.path.join(models_directory, 'dictionary.pkl')
        self.dictionary = Dictionary.load_from_file(path)
        self.modelIndexByWord = self.dictionary.word2idx
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self.load_models(os.path.join(models_directory, 'networks'),
                                         prefix='best_loss_model',
                                         device=device)
        self.thresholds_by_id = self.load_threshold(models_directory)
        model_list = sorted(self.models.keys())
        th_list = sorted(self.thresholds_by_id.keys())
        for i in model_list:
            if i not in th_list:
                logger.warning("no threshold for {0}".format(i))
                self.thresholds_by_id[i] = 0.5

    def create_networks(self, all_words, device):
        nets = dict()
        for k in all_words:
            model = nn.Sequential(
                nn.Linear(2048, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ).to(device)
            nets[k] = model
        return nets

    def get_parameters(self, nets):
        rez = []
        for k in nets:
            rez += nets[k].parameters()
        return rez

    def set_all_train(self, nets, is_train):
        for k in nets:
            nets[k].train(is_train)

    def load_models(self, path_to_models, prefix, device):
        list_of_files = glob.glob(path_to_models + "/" + prefix + "_*.pth")
        list_of_words = []
        for f in list_of_files:
            rez = int(re.findall("_(\d+)\.pth", f)[0])
            list_of_words.append(rez)

        nets = self.create_networks(list_of_words, device)

        for f, w in zip(list_of_files, list_of_words):
            nets[w].load_state_dict(torch.load(f, map_location='cpu'))

        return nets

    def load_threshold(self, directory):
        with open(os.path.join(directory, 'thresholds/best_th.json'), 'r') as f:
            return {int(k): float(v) for (k,v) in json.load(f).items()}

    def get_threshold_by_word(self, word):
        word_id = self.dictionary.word2idx[word]
        return self.thresholds_by_id[word_id]

    def get_words(self):
        result = set()
        for idx in self.models.keys():
            result.add(self.dictionary.idx2word[idx])
        return result


class SplitMultidnnRunner(NeuralNetworkRunner):

    def __init__(self, models_directory):
        self.nets_vocabulary = SplitNetsVocab(models_directory)

    def runNeuralNetwork(self, features, word):
        logger.debug("processing word {0}".format(word))
        try:
            model = self.nets_vocabulary.get_model_by_word(word)
        except KeyError as e:
            logger.debug("No model for word {0}".format(word))
            model = None

        if model is None:
            logger.debug('no model found, return FALSE')
            raise NoModelException("No model for word: {0}".format(word))

        # todo: threshold should be part of the model
        threshold = self.nets_vocabulary.get_threshold_by_word(word)
        # use threshold = 0.5 + delta
        # instead of f(x) > 0.5 + delta
        # we will check for f(x) - delta > 0.5, where delta = threshold - 0.5
        delta = threshold - 0.5
        result = model(torch.Tensor(features))
        # take max to keep values in valid range (0, 1)
        return max(torch.tensor(0.0), result - delta)
