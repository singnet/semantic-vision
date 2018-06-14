package org.opencog.vqa;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.stream.Stream;

import relex.ParsedSentence;
import relex.RelationExtractor;
import relex.Sentence;

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
            linesStream.map(QuestionRecord::load)
                    .map(this::parseQuestion)
                    .map(parsedRecord -> parsedRecord.getRecord().save())
                    .parallel()
                    .forEach(System.out::println);
        } finally {
            linesStream.close();
        }
    }

    private ParsedQuestion parseQuestion(QuestionRecord record) {
        Sentence sentence = relationExtractor.processSentence(record.getQuestion());
        ParsedSentence parsedSentence = sentence.getParses().get(0);
        
        RelexFormulaBuildingVisitor relexVisitor = new RelexFormulaBuildingVisitor();
        parsedSentence.foreach(relexVisitor);
        RelexFormula relexFormula = relexVisitor.getRelexFormula();
        
        QuestionRecord recordWithFormula = record.toBuilder()
                .shortFormula(relexFormula.getShortFormula())
                .fullFormula(relexFormula.getFullFormula())
                .build();
        
        return new ParsedQuestion(recordWithFormula, sentence);
    }

    private static class ParsedQuestion {

        private final QuestionRecord record;
        private final Sentence sentence;

        public ParsedQuestion(QuestionRecord record, Sentence sentence) {
            this.record = record;
            this.sentence = sentence;
        }

        public QuestionRecord getRecord() {
            return record;
        }

        public Sentence getSentence() {
            return sentence;
        }
    }

    private static void handleException(Exception e) {
        e.printStackTrace();
    }

}
