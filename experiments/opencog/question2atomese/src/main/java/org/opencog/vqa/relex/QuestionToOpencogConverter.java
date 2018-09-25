package org.opencog.vqa.relex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import relex.ParsedSentence;
import relex.RelationExtractor;
import relex.Sentence;

import com.google.common.collect.ImmutableList;

public class QuestionToOpencogConverter {

    private final RelationExtractor relationExtractor;
    private final List<ToQueryConverter> toQueryConverters;
    
    public QuestionToOpencogConverter() {
        this.relationExtractor = new RelationExtractor();
        this.relationExtractor.setMaxParses(1);
        this.toQueryConverters = ImmutableList.of(
                new YesNoPredadjToSchemeQueryConverter(),
                new WhatOtherDetObjSubjToSchemeQueryConverter()
            );
    }
    
    public RelexFormula parseQuestion(String question) {
        Sentence sentence = relationExtractor.processSentence(question);
        ParsedSentence parsedSentence = sentence.getParses().get(0);
        
        RelexFormulaBuildingVisitor relexVisitor = new RelexFormulaBuildingVisitor(parsedSentence);
        parsedSentence.foreach(relexVisitor);
        return relexVisitor.getRelexFormula();
    }

    public String convertToOpencogScheme(RelexFormula formula) {
        for (ToQueryConverter converter : toQueryConverters) {
            if (converter.isApplicable(formula)) {
                return converter.getSchemeQuery(formula);
            }
        }
        
        return null;
    }
    
    public Map<String, String> getFormulaQuestionMap(){
    	HashMap<String, String> result = new HashMap<String, String>();
    	for (ToQueryConverter converter : toQueryConverters) {
    		result.put(converter.getFullFormula(), converter.getQuestionType());
    	}
    	return result;
    }
}
