package org.opencog.vqa;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.Optional;
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
    private static final String OPTION_ATOMSPACE = "atomspace";
    
    private final BufferedReader bufferedReader;
    private final PrintWriter printWriter;
    private final PrintWriter atomspaceWriter;
    
    private final QuestionToOpencogConverter questionToOpencogConverter;

    private QuestionToOpencogApp(InputStream inputStream,
            OutputStream outputStream,
            Optional<OutputStream> atomspaceWriter) {
        this.bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
        this.printWriter = new PrintWriter(outputStream);
        if (atomspaceWriter.isPresent()) {
            this.atomspaceWriter = new PrintWriter(atomspaceWriter.get());
        } else {
            this.atomspaceWriter = null;
        }
        this.questionToOpencogConverter = new QuestionToOpencogConverter();
    }

    public static void main(String args[]) {
        Options options = new Options();
        InputStream inputStream = null;
        OutputStream outputStream = null;
        OutputStream atomspaceStream = null;
        try {
            options.addOption("i", OPTION_INPUT, true, "input filename, stdin if not provided");
            options.addOption("o", OPTION_OUTPUT, true, "output filename, stdout if not provided");
            options.addOption("a", OPTION_OUTPUT, true, "filename for atomspace which is calculated from questions");
            
            CommandLineParser argsParser = new DefaultParser();
            CommandLine commandLine = argsParser.parse(options, args);

            inputStream = commandLine.hasOption(OPTION_INPUT) 
                    ? new FileInputStream(commandLine.getOptionValue(OPTION_INPUT))
                    : System.in;
            outputStream = commandLine.hasOption(OPTION_OUTPUT) 
                    ? new FileOutputStream(commandLine.getOptionValue(OPTION_OUTPUT))
                    : System.out;
            atomspaceStream = commandLine.hasOption(OPTION_ATOMSPACE)
                    ? new FileOutputStream(commandLine.getOptionValue(OPTION_ATOMSPACE))
                    : null;
            new QuestionToOpencogApp(inputStream, outputStream, Optional.ofNullable(atomspaceStream)).run();
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("QuestionToOpencogApp", options);
        } catch (Exception e) {
            handleException(e);
        } finally {
            closeStream(inputStream, System.in);
            closeStream(outputStream, System.out);
            closeStream(atomspaceStream, System.out);
        }
    }

    private static void closeStream(Closeable stream, Closeable dflt) {
        if (stream != null && stream != dflt) {
            try {
                stream.close();
            } catch (IOException e) {
                handleException(e);
            }
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
                    .parallel()
                    .forEach(parsedRecord -> {
                        printWriter.println(parsedRecord.getRecord().save());
                    });
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
