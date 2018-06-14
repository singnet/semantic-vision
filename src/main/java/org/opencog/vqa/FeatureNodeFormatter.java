package org.opencog.vqa;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import relex.feature.FeatureNode;

/**
 * Utility class to format full content of FeatureNode.
 */
public class FeatureNodeFormatter {


    /**
     * Format full content of FeatureNode as text tree. Assigns unique id to each node and doesn't print repeated 
     * nodes, prints their id instead.  
     * 
     * @param featureNode node to start printing from
     * @return text tree as string
     */
    public static String formatAsTree(FeatureNode featureNode) {
        return featureTreeToString(featureNode, "", new Visited());
    }

    private static String featureTreeToString(FeatureNode featureNode, String alignment, Visited visited) {
        if (visited.contains(featureNode)) {
            return "<" + visited.get(featureNode) + ">";
        }

        if (featureNode.isValued()) {
            return featureNode.getValue();
        }

        int id = visited.put(featureNode);

        return "<" + id + ">: \n" + alignment + featureNode.getFeatureNames().stream().map(
                feature -> feature + ": " + featureTreeToString(featureNode.get(feature), alignment + " ", visited))
                .collect(Collectors.joining("\n" + alignment));
    }

    private static class Visited {

        private final Map<FeatureNode, Integer> visited = new HashMap<>();
        private int nextId = 0;

        public boolean contains(FeatureNode featureNode) {
            return visited.containsKey(featureNode);
        }

        public int get(FeatureNode featureNode) {
            return visited.get(featureNode);
        }

        public int put(FeatureNode featureNode) {
            int id = nextId++;
            visited.put(featureNode, id);
            return id;
        }
    }
    
}
