package org.opencog.vqa.relex;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class QuestionToOpencogConverterTest {

    private QuestionToOpencogConverter questionToOpencogConverter;
    
    @Before
    public void setUp() {
        questionToOpencogConverter = new QuestionToOpencogConverter();
    }
    
    @Test
    public void test_YesNoPredadj_IsTheRoomDark() {
        RelexFormula formula = questionToOpencogConverter.parseQuestion("Is the room dark?");
        String scheme = questionToOpencogConverter.convertToOpencogScheme(formula);
        System.out.println(scheme);
        Assert.assertEquals("(BindLink\n" +
                            "  (TypedVariableLink (VariableNode \"$X\") (TypeNode \"ConceptNode\"))\n" +
                            "  (AndLink\n" +
                            "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"BoundingBox\"))\n" +
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"room\")) )\n" + 
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"dark\")) )\n" +
                            "  )\n" +
                            "  (AndLink\n" +
                            "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"BoundingBox\"))\n" +
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"room\")) )\n" +
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"dark\")) )\n" +
                            "  )\n" +
                            ")\n"
                            , scheme);
    }
    
    @Test
    public void test_YesNoPredadj_IsTheRoomDarkURE() {
        RelexFormula formula = questionToOpencogConverter.parseQuestion("Is the room dark?");
        String scheme = questionToOpencogConverter.convertToOpencogSchemeURE(formula);
        System.out.println(scheme);
        Assert.assertEquals("(conj-bc (AndLink\n" +
			                "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"BoundingBox\"))\n" +
			                "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"room\")) )\n" +
			                "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"dark\")) )\n" +
			                "  )\n )", scheme);
    }

    @Test
    public void test_OtherDetObjSubj_WhatColorIsTheSky() {
        RelexFormula formula = questionToOpencogConverter.parseQuestion("What color is the sky?");
        String scheme = questionToOpencogConverter.convertToOpencogScheme(formula);
        Assert.assertEquals("(BindLink\n" +
                            "  (VariableList\n" +
                            "    (TypedVariableLink (VariableNode \"$B\") (TypeNode \"ConceptNode\"))\n" +
                            "    (TypedVariableLink (VariableNode \"$X\") (TypeNode \"ConceptNode\"))\n" +
                            "  )\n" +
                            "  (AndLink\n" +
                            "    (InheritanceLink (VariableNode \"$B\") (ConceptNode \"BoundingBox\"))\n" +
                            "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"color\"))\n" +
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$B\") (ConceptNode \"sky\")) )\n" +
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$B\") (VariableNode \"$X\")) )\n" +
                            "  )\n" +
                            "  (AndLink\n" +
                            "    (InheritanceLink (VariableNode \"$B\") (ConceptNode \"BoundingBox\"))\n" +
                            "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"color\"))\n" +
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$B\") (ConceptNode \"sky\")) )\n" +
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$B\") (VariableNode \"$X\")) )\n" +
                            "  )\n" +
                            ")\n"
                            , scheme);
    }

    @Test
    public void test_OtherDetObjSubj_WhatColorIsTheSkyURE() {
        RelexFormula formula = questionToOpencogConverter.parseQuestion("What color is the sky?");
        String scheme = questionToOpencogConverter.convertToOpencogSchemeURE(formula);
        Assert.assertEquals("(conj-bc   (AndLink\n" +
                            "    (InheritanceLink (VariableNode \"$B\") (ConceptNode \"BoundingBox\"))\n" +
                            "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"color\"))\n" +
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$B\") (ConceptNode \"sky\")) )\n" + 
                            "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$B\") (VariableNode \"$X\")) )\n" +
                            "  )\n" +
                            ")\n"
                            , scheme);
    }

    @Test
	public void test_ParseQuestionWithType() {
	    String question = "Are zebras fat?";
	    QuestionToOpencogConverter.ParsedQuestion parsedQuestion = questionToOpencogConverter.parseQuestionAndType(question);
	    Assert.assertEquals(parsedQuestion.questionType, "yes/no");
	    Assert.assertEquals(parsedQuestion.relexFormula.getFullFormula(), "_predadj(A, B)");
	}
}
