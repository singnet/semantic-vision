package org.opencog.vqa;

import java.util.ArrayList;
import java.util.List;

import relex.feature.FeatureNode;

class RelexArgument {
    
    private final FeatureNode featureNode;
    private final List<RelexPredicate> relations;
    private final String variableName;

    public RelexArgument(FeatureNode featureNode, String variableName) {
        this.featureNode = featureNode;
        this.relations = new ArrayList<>();
        this.variableName = variableName;
    }

    public String getVariableName() {
        return variableName;
    }

    @Override
    public String toString() {
        if (featureNode.get("name") == null) {
            return "XXXX";
        }
        return featureNode.get("name").getValue();
    }

    int getNumberOfUsages() {
        return relations.size();
    }

    void addRelation(RelexPredicate relation) {
        relations.add(relation);
    }

}