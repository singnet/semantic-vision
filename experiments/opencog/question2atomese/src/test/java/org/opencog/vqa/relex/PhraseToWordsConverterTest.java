package org.opencog.vqa.relex;

import java.util.Set;

import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.ImmutableSet;

public class PhraseToWordsConverterTest {
    
    @Test
    public void test() {
        Set<String> words = new PhraseToWordsConverter().parsePhrase("banana, orange and apples\n");
        Assert.assertEquals(ImmutableSet.of("banana", "orange", "apple"), words);
    }
}
