package org.opencog.vqa.relex;

import static com.google.common.base.Preconditions.checkArgument;

import relex.feature.FeatureNode;

class RelexUtils {

    public static String getFeatureNodeName(FeatureNode featureNode) {
        FeatureNode name = featureNode.get("name");
        if (name == null) {
            checkArgument(name != null, "FeatureNode doesn't have name attribute set");
        }
        
        return name.getValue();
    }


}
