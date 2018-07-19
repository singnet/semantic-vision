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
        InputStream inputStream = new ByteArrayInputStream("458752002::other::What color is the players shirt?::458752::orange\n".getBytes());
        ByteArrayOutputStream atomspaceStream = new ByteArrayOutputStream();
        
        QuestionToOpencogApp app = new QuestionToOpencogApp(inputStream, System.out, Optional.of(atomspaceStream));
        app.run();
        
        assertEquals("(InheritanceLink (ConceptNode \"orange\") (ConceptNode \"color\"))\n", atomspaceStream.toString());
    }
    
}
