package atomspace.storage.memory;

import atomspace.storage.ASAbstractRawLinkTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.RawLink;
import atomspace.storage.RawNode;
import org.junit.Assert;
import org.junit.Test;

public class ASMemoryRawLinkTest extends ASAbstractRawLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }

    @Test
    public void testSame() throws Exception {

        final RawLink link1 = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        final RawLink link2 = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        testAtomspaceTransaction(as ->
                Assert.assertSame(as.get(link1), as.get(link2)));
    }
}
