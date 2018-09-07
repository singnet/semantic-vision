package org.opencog.vqa;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.Optional;

import org.junit.Test;

public class QuestionToOpencogAppTest {

    @Test
    public void test_GetFactFromWhatQuestion() {
        InputStream inputStream = inputStreamFromString("458752002::other::What color is the players shirt?::458752::orange");
        ByteArrayOutputStream atomspaceStream = new ByteArrayOutputStream();
        
        new QuestionToOpencogApp(inputStream, System.out, Optional.of(atomspaceStream)).run();
        
        assertEquals("(InheritanceLink (ConceptNode \"orange\") (ConceptNode \"color\"))\n", atomspaceStream.toString());
    }

    @Test
    public void test_GetFewFactsFromOneWhatQuestion() {
        InputStream inputStream = inputStreamFromString("579057010::other::What fruits are these?::579057::banana, orange and apples");
        ByteArrayOutputStream atomspaceStream = new ByteArrayOutputStream();
        
        new QuestionToOpencogApp(inputStream, System.out, Optional.of(atomspaceStream)).run();
        
        assertEquals("(InheritanceLink (ConceptNode \"banana\") (ConceptNode \"fruit\"))\n"
                + "(InheritanceLink (ConceptNode \"orange\") (ConceptNode \"fruit\"))\n"
                + "(InheritanceLink (ConceptNode \"apple\") (ConceptNode \"fruit\"))\n", atomspaceStream.toString());
    }
    
    @Test
    public void test_IgnorePronoun() {
        InputStream inputStream = inputStreamFromString("418489000::other::What is this color of the shirt?::418489::red");
        ByteArrayOutputStream atomspaceStream = new ByteArrayOutputStream();
        
        new QuestionToOpencogApp(inputStream, System.out, Optional.of(atomspaceStream)).run();
        
        assertEquals("", atomspaceStream.toString());
    }

    @Test
    public void test_ParseValidSentence() {
        test_ParseSentence("196608003::yes/no::Is the laptop on?::196608::no::None::None",
                "196608003::yes/no::Is the laptop on?::196608::no::_subj(A, B)::_subj(be_on, laptop)");
    }

    @Test
    public void test_ParseInvalidSentence() {
        test_ParseSentence("158335000::number::How many cakes on in her hand?::158335::2::None::None",
                "158335000::number::How many cakes on in her hand?::158335::2::SKIPPED::None");
    }

    private void test_ParseSentence(String sentence, String parsedSentence) {
        InputStream inputStream = inputStreamFromString(sentence);
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

        new QuestionToOpencogApp(inputStream, outputStream, Optional.empty()).run();

        String record = outputStream.toString();
        System.out.printf("record: %s%n", record);
        assertEquals(parsedSentence, outputStream.toString().trim());
    }

    private static ByteArrayInputStream inputStreamFromString(String string) {
        return new ByteArrayInputStream(string.getBytes());
    }

}
