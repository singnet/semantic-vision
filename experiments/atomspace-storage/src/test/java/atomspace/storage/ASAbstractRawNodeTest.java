package atomspace.storage;

import org.junit.Assert;
import org.junit.Test;

public abstract class ASAbstractRawNodeTest extends ASAbstractTest {

    @Test
    public void testNotNull() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertNotNull(as.get(new RawNode("Node", "value"))));
    }

    @Test
    public void testEquals() throws Exception {

        final RawNode node1 = new RawNode("Node", "value");
        final RawNode node2 = new RawNode("Node", "value");
        final RawNode node3 = new RawNode("Node1", "value");
        final RawNode node4 = new RawNode("Node", "value1");

        Assert.assertEquals(node1, node2);
        Assert.assertNotEquals(node1, node3);
        Assert.assertNotEquals(node1, node4);

        testAtomspaceTransaction(as -> {
            Assert.assertEquals(as.get(node1), as.get(node2));
            Assert.assertNotEquals(as.get(node1), as.get(node3));
            Assert.assertNotEquals(as.get(node1), as.get(node4));
        });
    }


    @Test
    public void testHashcode() throws Exception {

        final RawNode node1 = new RawNode("Node", "value");
        final RawNode node2 = new RawNode("Node", "value");
        final RawNode node3 = new RawNode("Node1", "value");
        final RawNode node4 = new RawNode("Node", "value1");

        Assert.assertEquals(node1.hashCode(), node2.hashCode());
        Assert.assertNotEquals(node1.hashCode(), node3.hashCode());
        Assert.assertNotEquals(node1.hashCode(), node4.hashCode());

        testAtomspaceTransaction(as -> {
            Assert.assertEquals(as.get(node1).hashCode(), as.get(node2).hashCode());
            Assert.assertNotEquals(as.get(node1).hashCode(), as.get(node3).hashCode());
            Assert.assertNotEquals(as.get(node1).hashCode(), as.get(node4).hashCode());
        });
    }

    @Test
    public void testToString() throws Exception {

        final RawNode node = new RawNode("Node", "value");
        Assert.assertEquals("Node('value')", node.toString());

        testAtomspaceTransaction(as ->
                Assert.assertEquals("Node('value')", as.get(node).toString()));
    }
}
