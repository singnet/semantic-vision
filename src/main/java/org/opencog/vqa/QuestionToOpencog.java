package org.opencog.vqa;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import relex.ParsedSentence;
import relex.RelationExtractor;
import relex.Sentence;
import relex.feature.FeatureNode;
import relex.feature.RelationCallback;

public class QuestionToOpencog {

    private final BufferedReader bufferedReader;
    private final RelationExtractor relationExtractor;

    private QuestionToOpencog(InputStream inputStream) {
        this.bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
        this.relationExtractor = new RelationExtractor();
        this.relationExtractor.setMaxParses(1);
    }

    public static void main(String args[]) {
        try {

            String filename = args[0];
            new QuestionToOpencog(new FileInputStream(filename)).run();

        } catch (Exception e) {
            handleException(e);
        }
    }

    private void run() {
        Stream<String> linesStream = bufferedReader.lines();
//        Stream<String> linesStream = Stream.of("0:yes/no:Is the room messy?:0");
        try {
            linesStream.map(Record::load)
                    .map(this::parseQuestion)
                    .map(parsedRecord -> parsedRecord.getRecord().save())
                    .parallel()
                    .forEach(System.out::println);
        } finally {
            linesStream.close();
        }
    }

    private ParsedRecord parseQuestion(Record record) {
        Sentence sentence = relationExtractor.processSentence(record.getQuestion());
        
        ParsedSentence parsedSentence = sentence.getParses().get(0);
        RelationCollectingVisitor relationsCollector = new RelationCollectingVisitor();
        parsedSentence.foreach(relationsCollector);
        Record recordWithFormula = record.toBuilder()
                .shortFormula(relationsCollector.getShortFormula())
                .formula(relationsCollector.getFormula())
                .build();
        
        return new ParsedRecord(recordWithFormula, sentence);
    }

    private static class ParsedRecord {

        private final Record record;
        private final Sentence sentence;

        public ParsedRecord(Record record, Sentence sentence) {
            this.record = record;
            this.sentence = sentence;
        }

        public Record getRecord() {
            return record;
        }

        public Sentence getSentence() {
            return sentence;
        }
    }

    private static class RelationCollectingVisitor implements RelationCallback {

        private final Map<FeatureNode, RelationArgument> argumentCache = new HashMap<>();
        private char nextVariableName = 'A';
        
        private final List<Relation> relations = new ArrayList<>();

        @Override
        public Boolean BinaryHeadCB(FeatureNode arg0) {
            return false;
        }

        @Override
        public Boolean BinaryRelationCB(String relation, FeatureNode first, FeatureNode second) {
            RelationArgument firstArg = toArgument(first);
            RelationArgument secondArg = toArgument(second);
            relations.add(new Relation(relation, firstArg, secondArg));
            return false;
        }

        private RelationArgument toArgument(FeatureNode featureNode) {
            return argumentCache.computeIfAbsent(featureNode,
                    fn -> new RelationArgument(fn, getNextVariableName()));
        }

        private String getNextVariableName() {
            return String.valueOf(nextVariableName++);
        }

        @Override
        public Boolean UnaryRelationCB(FeatureNode arg0, String arg1) {
            return false;
        }

        public String getFormula() {
            relations.sort(Comparator.naturalOrder());
            return relations.stream().map(fn -> fn.toFormula()).collect(Collectors.joining(";"));
        }

        public String getShortFormula() {
            relations.sort(Comparator.naturalOrder());
            return relations.stream().map(fn -> fn.toShortFormula()).collect(Collectors.joining(";"));
        }
    }

    private static class RelationArgument {
        private final FeatureNode featureNode;
        private final List<Relation> relations;
        private final String variableName;

        public RelationArgument(FeatureNode featureNode, String variableName) {
            this.featureNode = featureNode;
            this.relations = new ArrayList<>();
            this.variableName = variableName;
        }

        int getNumberOfUsages() {
            return relations.size();
        }

        @Override
        public String toString() {
            if (featureNode.get("name") == null) {
                return "XXXX";
            }
            return featureNode.get("name").getValue();
        }

        void addRelation(Relation relation) {
            relations.add(relation);
        }

        public String getVariableName() {
            return variableName;
        }
    }

    private static class Relation implements Comparable<Relation> {
        private final String name;
        private final List<RelationArgument> arguments;

        public Relation(String name, RelationArgument firstArg, RelationArgument secondArg) {
            this.name = name;
            this.arguments = List.of(firstArg, secondArg);
            firstArg.addRelation(this);
            secondArg.addRelation(this);
        }

        @Override
        public int compareTo(Relation other) {
            if (getNumberOfArgumentUsages() != other.getNumberOfArgumentUsages()) {
                return getNumberOfArgumentUsages() - other.getNumberOfArgumentUsages();
            }
            return name.compareTo(other.name);
        }

        private int getNumberOfArgumentUsages() {
            return arguments.stream().collect(Collectors.summingInt(RelationArgument::getNumberOfUsages));
        }

        public String toFormula() {
            return name + "(" + arguments.stream().map(fn -> fn.getVariableName()).collect(Collectors.joining(", "))
                    + ")";
        }

        public String toShortFormula() {
            return name + "()";
        }

        @Override
        public String toString() {
            return name + "(" + arguments.stream().map(fn -> fn.toString()).collect(Collectors.joining(", ")) + ")";
        }
    }


    private static void handleException(Exception e) {
        e.printStackTrace();
    }

}
