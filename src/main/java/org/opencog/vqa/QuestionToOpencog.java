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
    
    private static void handleException(Exception e) {
        e.printStackTrace();
    }

    private void run() {
        Stream<String> linesStream = bufferedReader.lines();
//        Stream<String> linesStream = Stream.of("0:yes/no:Is the room messy?:0");
        try {
            linesStream.map(QuestionRecord::load)
                    .map(this::parseQuestion)
                    .map(parsedRecord -> parsedRecord.getRecord().save())
//                    .map(this::convertToOpencogSchema)
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
        
        return new ParsedQuestion(recordWithFormula, relexFormula);
    }

    private static class ParsedQuestion {

        private final QuestionRecord record;
        private final RelexFormula relexFormula;

        public ParsedQuestion(QuestionRecord record, RelexFormula relexFormula) {
            this.record = record;
            this.relexFormula = relexFormula;
        }

        public QuestionRecord getRecord() {
            return record;
        }

        public RelexFormula getRelexFormula() {
            return relexFormula;
        }
    }
    
    private String convertToOpencogSchema(ParsedQuestion parsedQuestion) {
        RelexFormula formula = parsedQuestion.getRelexFormula();
        
        if (formula.getShortFormula().equals("_predadj(A, B)")) {
            
        }
        
        return "<not supported>";
    }

}
