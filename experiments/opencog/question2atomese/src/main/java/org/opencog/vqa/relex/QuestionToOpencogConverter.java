package org.opencog.vqa.relex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import relex.ParsedSentence;
import relex.RelationExtractor;
import relex.Sentence;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;

public class QuestionToOpencogConverter {

	@VisibleForTesting
	public class ParsedQuestion {
		public final RelexFormula relexFormula;
		public final String questionType;
		public ParsedQuestion(RelexFormula relexFormula, String questionType) {
			this.relexFormula = relexFormula;
			this.questionType = questionType;
		}
	}

    private final RelationExtractor relationExtractor;
    private final List<ToQueryConverter> toQueryConverters;
    private final Map<String, String> formulaQuestionTypeMap;

    public QuestionToOpencogConverter() {
        this.relationExtractor = new RelationExtractor();
        this.relationExtractor.setMaxParses(1);
        this.toQueryConverters = ImmutableList.of(
                new YesNoPredadjToSchemeQueryConverter(),
                new WhatOtherDetObjSubjToSchemeQueryConverter()
            );
        this.formulaQuestionTypeMap = this.computeFormulaQuestionMap();
    }
    
    public RelexFormula parseQuestion(String question) {
        Sentence sentence = relationExtractor.processSentence(question);
        ParsedSentence parsedSentence = sentence.getParses().get(0);
        
        RelexFormulaBuildingVisitor relexVisitor = new RelexFormulaBuildingVisitor(parsedSentence);
        parsedSentence.foreach(relexVisitor);
        return relexVisitor.getRelexFormula();
    }

	public ParsedQuestion parseQuestionAndType(String question) {
		RelexFormula questionFormula = parseQuestion(question);
		String questionType = this.formulaQuestionTypeMap.get(questionFormula.getFullFormula());
		return new ParsedQuestion(questionFormula, questionType);
	}

    public String convertToOpencogScheme(RelexFormula formula) {
        for (ToQueryConverter converter : toQueryConverters) {
            if (converter.isApplicable(formula)) {
                return converter.getSchemeQuery(formula);
            }
        }
        
        return null;
    }

    public String convertToOpencogSchemePM(RelexFormula formula) {
        for (ToQueryConverter converter : toQueryConverters) {
            if (converter.isApplicable(formula)) {
                return converter.getSchemeQueryPM(formula);
            }
        }

        return null;
    }

    public String convertToOpencogSchemeURE(RelexFormula formula) {
        for (ToQueryConverter converter : toQueryConverters) {
            if (converter.isApplicable(formula)) {
                return converter.getSchemeQueryURE(formula);
            }
        }

        return null;
    }

    private Map<String, String> computeFormulaQuestionMap(){
    	HashMap<String, String> result = new HashMap<String, String>();
    	for (ToQueryConverter converter : toQueryConverters) {
    		result.put(converter.getFullFormula(), converter.getQuestionType());
    	}
    	return result;
    }
}
