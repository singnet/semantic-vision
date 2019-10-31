package atomspace.storage;

import org.junit.Assert;
import org.junit.Test;

public abstract class ASAbstractLinkTest extends ASAbstractTest {

    @Test
    public void testNotNull() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertNotNull(
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value2"))));
    }

    @Test
    public void testEquals() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertEquals(
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value2")),
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value2"))));
    }

    @Test
    public void testIdEquals() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertEquals(
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value2")).getId(),
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value2")).getId()));
    }

    @Test
    public void testHashcode() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertEquals(
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value2")).hashCode(),
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value2")).hashCode()));
    }

    @Test
    public void testToString() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertEquals(
                        "Link(Node1('value1'),Node2('value2'))",
                        as.get("Link",
                                as.get("Node1", "value1"),
                                as.get("Node2", "value2")).toString()));
    }

    @Test
    public void testZeroArity() throws Exception {

        testAtomspaceTransaction(as -> {
            ASLink link = as.get("Link");
            Assert.assertEquals(link, as.get("Link"));
        });
    }
}
