package atomspace.storage.memory;

import atomspace.storage.ASAbstractLinkTest;
import atomspace.storage.AtomspaceStorage;
import org.junit.Assert;
import org.junit.Test;

public class ASMemoryLinkTest extends ASAbstractLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }

    @Test
    public void testSame() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertSame(
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value-2")),
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value-2"))));
    }
}
