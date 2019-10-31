package atomspace.storage.memory;

import atomspace.storage.ASAbstractNodeTest;
import atomspace.storage.AtomspaceStorage;
import org.junit.Assert;
import org.junit.Test;

public class ASMemoryNodeTest extends ASAbstractNodeTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }

    @Test
    public void testSame() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertSame(
                        as.get("Node", "value"),
                        as.get("Node", "value")));
    }
}
