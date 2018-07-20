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
    public void test_PredadjQuestionConverter() {
        RelexFormula formula = questionToOpencogConverter.parseQuestion("Is the room dark?");
        String scheme = questionToOpencogConverter.convertToOpencogScheme(formula);
        Assert.assertEquals("(SatisfactionLink " + 
                            "(TypedVariableLink (VariableNode \"$X\") (TypeNode \"ConceptNode\")) " +
                            "(AndLink " +
                            "(InheritanceLink (VariableNode \"$X\") (ConceptNode \"BoundingBox\")) " +
                            "(EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"room\")) )" + 
                            "(EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"dark\")) )" +
                            ") )"
                            , scheme);
    }
    
}
