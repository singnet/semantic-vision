package atomspace.storage;

import org.junit.Assert;
import org.junit.Test;

public abstract class ASAbstractRegressionTest extends ASAbstractTest {

    /**
     * The top level link has the same type 'Link1' as its child link.
     * For some reason the backing storage may return
     * the child link instead of the top level one.
     */
    @Test
    public void testTopLevelLinkWithTheSameType() throws Exception {

        final RawLink rawLink = new RawLink("Link1",
                new RawLink("Link1",
                        new RawNode("Node1", "value1")),
                new RawLink("Link2",
                        new RawNode("Node2", "value2")));

        testAtomspaceTransaction(as -> {

            ASLink link = as.get(rawLink);
            ASOutgoingList outgoingList = link.getOutgoingList();
            Assert.assertEquals(2, outgoingList.getArity(as));
            Assert.assertNotNull(outgoingList.getAtom(as, 0));
            Assert.assertNotNull(outgoingList.getAtom(as, 1));
        });
    }
}
