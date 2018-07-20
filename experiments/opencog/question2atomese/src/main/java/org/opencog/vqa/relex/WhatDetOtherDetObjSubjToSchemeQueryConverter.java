package org.opencog.vqa.relex;

import relex.feature.FeatureNode;
import relex.feature.RelationCallback;

public class WhatDetOtherDetObjSubjToSchemeQueryConverter implements ToQueryConverter {

    @Override
    public boolean isApplicable(RelexFormula formula) {
        return formula.getFullFormula().equals("_det(A, B);_obj(C, D);_subj(C, A)");
    }

    @Override
    public String getSchemeQuery(RelexFormula relexFormula) {
        RelexVisitor visitor = new RelexVisitor();
        relexFormula.getRelexSentence().foreach(visitor);
        return String.format("(BindLink\n" + 
                "  (VariableList\n" +
                "    (TypedVariableLink (VariableNode \"$B\") (TypeNode \"ConceptNode\"))\n" +
                "    (TypedVariableLink (VariableNode \"$X\") (TypeNode \"ConceptNode\"))\n" +
                "  )\n" +
                "  (AndLink\n" +
                "    (InheritanceLink (VariableNode \"$B\") (ConceptNode \"BoundingBox\"))\n" +
                "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"%1$s\"))\n" +
                "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$B\") (ConceptNode \"%2$s\")) )\n" + 
                "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$B\") (VariableNode \"$X\")) )\n" +
                "  )\n" +
                "  (Variable \"$X\")\n" +
                ")\n"
                , visitor.attribute, visitor.object);
    }
    
    private static class RelexVisitor implements RelationCallback {
        
        String attribute;
        String object;
        
        @Override
        public Boolean UnaryRelationCB(FeatureNode node, String attrName) {
            return Boolean.FALSE;
        }

        @Override
        public Boolean BinaryRelationCB(String relation, FeatureNode srcNode, FeatureNode tgtNode) {
            if (relation.equals("_det")) {
                attribute = RelexUtils.getFeatureNodeName(srcNode);
            }
            if (relation.equals("_obj")) {
                object = RelexUtils.getFeatureNodeName(tgtNode);
            }
            return Boolean.FALSE;
        }

        @Override
        public Boolean BinaryHeadCB(FeatureNode from) {
            return Boolean.FALSE;
        }
    }

}
