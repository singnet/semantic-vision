package org.opencog.vqa.relex;

import java.util.HashMap;
import java.util.Map;

import relex.ParsedSentence;
import relex.feature.FeatureNode;
import relex.feature.RelationCallback;

class RelexFormulaBuildingVisitor implements RelationCallback {

    private final Map<FeatureNode, RelexArgument> argumentCache = new HashMap<>();
    private char nextVariableName = 'A';
    
    private final RelexFormula.RelexFormulaBuilder formulaBuilder;
    
    RelexFormulaBuildingVisitor(ParsedSentence relexSentence) {
        this.formulaBuilder = RelexFormula.builder().relexSentence(relexSentence);
    }

    @Override
    public Boolean BinaryHeadCB(FeatureNode arg0) {
        return false;
    }

    @Override
    public Boolean BinaryRelationCB(String relation, FeatureNode first, FeatureNode second) {
        RelexArgument firstArg = toArgument(first);
        RelexArgument secondArg = toArgument(second);
        formulaBuilder.predicate(new RelexPredicate(relation, firstArg, secondArg));
        return false;
    }

    @Override
    public Boolean UnaryRelationCB(FeatureNode arg0, String arg1) {
        return false;
    }

    public RelexFormula getRelexFormula() {
        return formulaBuilder.build();
    }
    
    private RelexArgument toArgument(FeatureNode featureNode) {
        return argumentCache.computeIfAbsent(featureNode,
                fn -> new RelexArgument(fn, getNextVariableName()));
    }

    private String getNextVariableName() {
        return String.valueOf(nextVariableName++);
    }

}