package org.opencog.vqa.relex;

import static com.google.common.base.Preconditions.checkArgument;

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
        return String.format("(SatisfactionLink " +
                "(TypedVariableLink (VariableNode \"$X\") (TypeNode \"ConceptNode\")) " +
                "(AndLink " +
                "(InheritanceLink (VariableNode \"$X\") (ConceptNode \"BoundingBox\")) " +
                "(EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"%1$s\")) )" +
                "(EvaluationLink (GroundedPredicateNode \"py:runNeuralNetwork\") (ListLink (VariableNode \"$X\") (ConceptNode \"%2$s\")) )" +
                ") )", visitor.object, visitor.state);
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
                object = getName(srcNode);
                state = getName(tgtNode);
            }
            return Boolean.FALSE;
        }

        private static String getName(FeatureNode featureNode) {
            FeatureNode name = featureNode.get("name");
            checkArgument(name != null, "FeatureNode doesn't have name attribute set");
            return name.getValue();
        }

        @Override
        public Boolean BinaryHeadCB(FeatureNode from) {
            return Boolean.FALSE;
        }
    }
    
}
