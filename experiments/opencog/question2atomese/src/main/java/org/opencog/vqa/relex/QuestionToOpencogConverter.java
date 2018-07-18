package org.opencog.vqa.relex;

import relex.ParsedSentence;
import relex.RelationExtractor;
import relex.Sentence;

public class QuestionToOpencogConverter {

    private final RelationExtractor relationExtractor;
    
    public QuestionToOpencogConverter() {
        this.relationExtractor = new RelationExtractor();
        this.relationExtractor.setMaxParses(1);
    }
    
    public RelexFormula parseQuestion(String question) {
        Sentence sentence = relationExtractor.processSentence(question);
        ParsedSentence parsedSentence = sentence.getParses().get(0);
        
        RelexFormulaBuildingVisitor relexVisitor = new RelexFormulaBuildingVisitor();
        parsedSentence.foreach(relexVisitor);
        return relexVisitor.getRelexFormula();
    }

    public String convertToOpencogSchema(RelexFormula formula) {
        if (formula.getFullFormula().equals("_predadj(A, B)")) {
            return formula.getAtomeseFormula();
        }
        
        return null;
    }
}
