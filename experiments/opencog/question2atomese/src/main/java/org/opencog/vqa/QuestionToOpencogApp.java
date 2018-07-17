package org.opencog.vqa;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.stream.Stream;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.opencog.vqa.relex.QuestionToOpencogConverter;
import org.opencog.vqa.relex.RelexFormula;

public class QuestionToOpencogApp {

    private static final String OPTION_INPUT = "input";
    private static final String OPTION_OUTPUT = "output";
    
    private final BufferedReader bufferedReader;
    private final PrintWriter printWriter;
    
    private final QuestionToOpencogConverter questionToOpencogConverter;

    private QuestionToOpencogApp(InputStream inputStream, OutputStream outputStream) {
        this.bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
        this.printWriter = new PrintWriter(outputStream);
        this.questionToOpencogConverter = new QuestionToOpencogConverter();
    }

    public static void main(String args[]) {
        Options options = new Options();
        try {
            options.addOption("i", OPTION_INPUT, true, "input filename, stdin if not provided");
            options.addOption("o", OPTION_OUTPUT, true, "output filename, stdout if not provided");
            
            CommandLineParser argsParser = new DefaultParser();
            CommandLine commandLine = argsParser.parse(options, args);

            InputStream inputStream = commandLine.hasOption(OPTION_INPUT) 
                    ? new FileInputStream(commandLine.getOptionValue(OPTION_INPUT))
                    : System.in;
            OutputStream outputStream = commandLine.hasOption(OPTION_OUTPUT) 
                    ? new FileOutputStream(commandLine.getOptionValue(OPTION_OUTPUT))
                    : System.out;
            new QuestionToOpencogApp(inputStream, outputStream).run();
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("QuestionToOpencogApp", options);
        } catch (Exception e) {
            handleException(e);
        }
    }
    
    private static void handleException(Exception e) {
        e.printStackTrace();
    }

    private void run() {
        Stream<String> linesStream = bufferedReader.lines();
//        Stream<String> linesStream = Stream.of("0::yes/no::Is the book a paperback?::0::none");
        try {
            linesStream.map(QuestionRecord::load)
                    .map(this::parseQuestion)
                    .map(parsedRecord -> parsedRecord.getRecord().save())
//                    .filter(parsedRecord -> parsedRecord.getRecord().getQuestionType().equals("yes/no"))
//                    .filter(parsedRecord -> parsedRecord.getRelexFormula().getFullFormula().equals("_predadj(A, B)"))
//                    .map(this::convertToOpencogSchema)
                    .parallel()
                    .forEach(printWriter::println);
        } finally {
            linesStream.close();
            printWriter.flush();
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
