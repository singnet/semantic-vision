package org.opencog.vqa.relex;

import relex.feature.FeatureNode;
import relex.feature.RelationCallback;

class YesNoPredadjToSchemeQueryConverter implements ToQueryConverter {

    @Override
    public boolean isApplicable(RelexFormula formula) {
        return formula.getFullFormula().equals("_predadj(A, B)");
    }

    @Override
    public String getSchemeQuery(RelexFormula relexFormula) {
        RelexVisitor visitor = new RelexVisitor();
        relexFormula.getRelexSentence().foreach(visitor);
        return String.format("(SatisfactionLink\n" +
                "  (TypedVariableLink (VariableNode \"$X\") (TypeNode \"ConceptNode\"))\n" +
                "  (AndLink\n" +
                "    (InheritanceLink (VariableNode \"$X\") (ConceptNode \"BoundingBox\"))\n" +
                "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"%1$s\")) )\n" +
                "    (EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"%2$s\")) )\n" +
                "  )\n" + 
                ")\n", visitor.object, visitor.state);
    }

    private static class RelexVisitor implements RelationCallback {
        
        String object;
        String state;
        
        @Override
        public Boolean UnaryRelationCB(FeatureNode node, String attrName) {
            return Boolean.FALSE;
        }

        @Override
        public Boolean BinaryRelationCB(String relation, FeatureNode srcNode, FeatureNode tgtNode) {
            if (relation.equals("_predadj")) {
                object = RelexUtils.getFeatureNodeName(srcNode);
                state = RelexUtils.getFeatureNodeName(tgtNode);
            }
            return Boolean.FALSE;
        }

        @Override
        public Boolean BinaryHeadCB(FeatureNode from) {
            return Boolean.FALSE;
        }
    }
    
}
