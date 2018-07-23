package org.opencog.vqa.relex;

import java.util.Set;

import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.ImmutableSet;

public class PhraseToWordsConverterTest {
    
    @Test
    public void test_GetListOfWordsFromAnswer() {
        Set<String> words = new PhraseToWordsConverter().parsePhrase("banana, orange and apples");
        Assert.assertEquals(ImmutableSet.of("banana", "orange", "apple"), words);
    }
    
    @Test
    public void test_GetListOfWordsFromSingleWordWhite() {
        Set<String> words = new PhraseToWordsConverter().parsePhrase("white");
        Assert.assertEquals(ImmutableSet.of("white"), words);
    }
}
