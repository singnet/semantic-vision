#!/usr/bin/python3
"""
Module for computing truth values for list of inheritance links
Truth values are computed from counts of the inheritance link,
to and from concept nodes.
"""

import re
import math
import argparse
from collections import defaultdict


def cached(function):
    cache = dict()
    def wrapped(self, *args):
        if args not in cache:
            cache[args] = function(self, *args)
        return cache[args]
    return wrapped


def tf(num_both, max_count_given_concept):
    """compute weight given count of (word, concept) pair and maximum count of among all words for given concept

    The returned value specifies how strongly given concept is associated with the word
    The fuction is based on "double normalization 0.5"
    from https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Term_frequency_2
    """
    return 0.5 + 0.5 * num_both / max_count_given_concept


def idf(num_both, max_concept_given_word):
    """compute weight given count of (word, concept) pair and maximum count of among all concepts for given word

    The returned value specifies how strongly given word is associated with the concept
    The function is base on "inverse document frequency smooth"
    from https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency_2
    The value is also normalized so that if num_both = max_concept_given_word the result will be 1.0
    """
    return math.log(1 + num_both / max_concept_given_word) / math.log(2)


def sigm(x, inv_weight=10):
    """Compute sigmoid of x/inv_weight"""
    return 1.0/(1.0 + math.exp(-x/inv_weight))


class TruthValueComputer:
    def __init__(self, lines):
        reg_comp = re.compile(".*ConceptNode\s\"(.*?)\".*ConceptNode\s\"(.*?)\".*")
        self.pairs = []
        for line in lines:
            match = reg_comp.match(line)
            if not match:
                print("Warning! Line was not parsed {0}".format(line))
                continue
            self.pairs.append(match.groups())

    @cached
    def count_word(self, word, pos):
        result = 0
        for item in self.pairs:
            if word == item[pos]:
                result += 1
        return result

    @cached
    def count_item(self, pair):
        return self.pairs.count(pair)

    @cached
    def count_adjacent(self, word, index):
        if index == 1:
            other_idx = 0
        else:
            other_idx = 1
        counts = defaultdict(int)
        for item in self.pairs:
            if word == item[index]:
                counts[item[other_idx]] += 1
        return max(counts.values()), sum(counts.values())

    def process(self, word, concept):
        num_both = self.count_item((word, concept))
        max_count_given_concept, sum_count_given_concept = self.count_adjacent(concept, 1)
        max_concept_given_word, sum_concept_given_word = self.count_adjacent(word, 0)
        strength = (idf(num_both, max_concept_given_word) + tf(num_both, max_count_given_concept)) / 2
        confidence = sigm(num_both)
        return strength, confidence

    def compute_weights(self):
        results = dict()
        for (word, concept) in self.pairs:
            if (word, concept) in results:
                continue
            results[word, concept] = self.process(word, concept)
        return results


def write_atomspace(pair_weights, output_file):
    template = '(InheritanceLink (stv {0} {1}) (ConceptNode "{2}") (ConceptNode "{3}"))\n'
    for (word, concept), (strength, confidence) in pair_weights.items():
        output_file.write(template.format(strength, confidence, word, concept))


def main():
    parser = make_parser()
    args = parser.parse_args()
    input_path = args.input
    out_path = args.output
    lines = open(input_path, 'r').readlines()
    tvc = TruthValueComputer(lines)
    with open(out_path, 'w') as f:
        write_atomspace(tvc.compute_weights(), f)
    print("done!")

def make_parser():
    parser = argparse.ArgumentParser(description='compute truth values')
    parser.add_argument('input', type=str,
                         help='source atomspace without truth values')
    parser.add_argument('output', type=str,
                         help='output path for atomspace with truth values')
    return parser


if __name__ == '__main__':
   main()

