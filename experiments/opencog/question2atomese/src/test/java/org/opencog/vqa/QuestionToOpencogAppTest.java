package org.opencog.vqa;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.Optional;

import org.junit.Test;
import org.opencog.vqa.relex.QuestionToOpencogConverter;
import org.opencog.vqa.relex.RelexFormula;

public class QuestionToOpencogAppTest {

    @Test
    public void test_GetFactFromWhatQuestion() {
        InputStream inputStream = new ByteArrayInputStream("458752002::other::What color is the players shirt?::458752::orange\n".getBytes());
        ByteArrayOutputStream atomspaceStream = new ByteArrayOutputStream();
        
        QuestionToOpencogApp app = new QuestionToOpencogApp(inputStream, System.out, Optional.of(atomspaceStream));
        app.run();
        
        assertEquals("(InheritanceLink (ConceptNode \"orange\") (ConceptNode \"color\"))\n", atomspaceStream.toString());
    }

    @Test
    public void test_GetFewFactsFromOneWhatQuestion() {
        InputStream inputStream = new ByteArrayInputStream("579057010::other::What fruits are these?::579057::banana, orange and apples\n".getBytes());
        ByteArrayOutputStream atomspaceStream = new ByteArrayOutputStream();
        
        QuestionToOpencogApp app = new QuestionToOpencogApp(inputStream, System.out, Optional.of(atomspaceStream));
        app.run();
        
        assertEquals("(InheritanceLink (ConceptNode \"banana\") (ConceptNode \"fruit\"))\n"
                + "(InheritanceLink (ConceptNode \"orange\") (ConceptNode \"fruit\"))\n"
                + "(InheritanceLink (ConceptNode \"apple\") (ConceptNode \"fruit\"))\n", atomspaceStream.toString());
    }

}
