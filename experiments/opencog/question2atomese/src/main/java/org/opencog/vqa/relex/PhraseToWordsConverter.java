package org.opencog.vqa.relex;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import relex.ParsedSentence;
import relex.RelationExtractor;
import relex.Sentence;
import relex.feature.FeatureNode;
import relex.feature.RelationCallback;

public class PhraseToWordsConverter {
    
    private final RelationExtractor relationExtractor;
    
    public PhraseToWordsConverter() {
        this.relationExtractor = new RelationExtractor();
        this.relationExtractor.setMaxParses(1);
    }
    
    public Set<String> parsePhrase(String phrase) {
        if (phrase.matches("\\w*")) {
            return Collections.singleton(phrase);
        }
        
        Sentence sentence = relationExtractor.processSentence(phrase);
        ParsedSentence parsedSentence = sentence.getParses().get(0);
        
        WordListBuildingVisitor relexVisitor = new WordListBuildingVisitor();
        parsedSentence.foreach(relexVisitor);
        return relexVisitor.getWords();
    }

    private static class WordListBuildingVisitor implements RelationCallback {

        private final Set<String> words = new HashSet<>();
        
        public Set<String> getWords() {
            return words;
        }

        @Override
        public Boolean UnaryRelationCB(FeatureNode node, String attrName) {
            if (node.get("pos") == null) {
                return false;
            }
            String pos = node.get("pos").getValue();
            if (pos.equals("punctuation") || pos.equals("conjunction")) {
                return false;
            }
            words.add(RelexUtils.getFeatureNodeName(node));
            return false;
        }

        @Override
        public Boolean BinaryRelationCB(String relation, FeatureNode srcNode, FeatureNode tgtNode) {
            return false;
        }

        @Override
        public Boolean BinaryHeadCB(FeatureNode from) {
            return false;
        }
    }
}
