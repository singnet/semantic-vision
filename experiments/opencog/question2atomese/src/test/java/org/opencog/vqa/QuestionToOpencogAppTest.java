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

    private static ByteArrayInputStream inputStreamFromString(String string) {
        return new ByteArrayInputStream(string.getBytes());
    }

}
