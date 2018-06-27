package org.opencog.vqa;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.stream.Stream;

import org.opencog.vqa.relex.QuestionToOpencogConverter;
import org.opencog.vqa.relex.RelexFormula;

public class QuestionToOpencogApp {

    private final BufferedReader bufferedReader;
    private final QuestionToOpencogConverter questionToOpencogConverter;

    private QuestionToOpencogApp(InputStream inputStream) {
        this.bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
        this.questionToOpencogConverter = new QuestionToOpencogConverter();
    }

    public static void main(String args[]) {
        try {

            String filename = args[0];
            new QuestionToOpencogApp(new FileInputStream(filename)).run();

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
//                    .filter(parsedRecord -> parsedRecord.getRecord().getQuestionType().equals("yes/no"))
//                    .filter(parsedRecord -> parsedRecord.getRelexFormula().getFullFormula().equals("_predadj(A, B)"))
//                    .map(this::convertToOpencogSchema)
                    .parallel()
                    .forEach(System.out::println);
        } finally {
            linesStream.close();
        }
    }

    private ParsedQuestion parseQuestion(QuestionRecord record) {
        RelexFormula relexFormula = questionToOpencogConverter.parseQuestion(record.getQuestion());
        
        QuestionRecord recordWithFormula = record.toBuilder()
                .shortFormula(relexFormula.getFullFormula())
                .fullFormula(relexFormula.getGroundedFormula())
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
        return questionToOpencogConverter.convertToOpencogSchema(parsedQuestion.getRelexFormula());
    }

}
