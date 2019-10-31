package atomspace.storage.memory;

import atomspace.storage.ASAbstractRawNodeTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.RawNode;
import org.junit.Assert;
import org.junit.Test;

public class ASMemoryRawNodeTest extends ASAbstractRawNodeTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }

    @Test
    public void testSame() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertSame(
                        as.get(new RawNode("Node", "value")),
                        as.get(new RawNode("Node", "value"))));
    }
}
