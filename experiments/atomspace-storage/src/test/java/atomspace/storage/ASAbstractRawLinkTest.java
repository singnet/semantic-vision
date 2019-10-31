package atomspace.storage;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public abstract class ASAbstractRawLinkTest extends ASAbstractTest {

    @Test
    public void testNotNull() throws Exception {

        final RawLink link = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        Assert.assertEquals(2, link.getArity());
        Assert.assertNotNull(link.getAtom(0));
        Assert.assertNotNull(link.getAtom(1));

        testAtomspaceTransaction(as ->
                Assert.assertNotNull(as.get(link)));
    }

    @Test
    public void testEquals() throws Exception {

        RawNode node1 = new RawNode("Node", "value1");
        RawNode node2 = new RawNode("Node", "value2");
        RawLink link = new RawLink("Link", node1, node2);

        Assert.assertEquals(2, link.getArity());
        Assert.assertEquals(node1, link.getAtom(0));
        Assert.assertEquals(node2, link.getAtom(1));

        final RawLink link1 = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        final RawLink link2 = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        final RawLink link3 = new RawLink("Link1",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        final RawLink link4 = new RawLink("Link",
                new RawNode("Node", "value3"),
                new RawNode("Node", "value3")
        );

        Assert.assertEquals(link1, link2);
        Assert.assertNotEquals(link1, link3);
        Assert.assertNotEquals(link1, link4);

        testAtomspaceTransaction(as ->
                Assert.assertEquals(as.get(link1), as.get(link2)));
    }

    @Test
    public void testArity() throws Exception {

        final RawLink link = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        Assert.assertEquals(2, link.getArity());

        testAtomspaceTransaction(as ->
                Assert.assertEquals(2, as.get(link).getOutgoingList().getArity(as)));
    }

    @Test
    public void testIdEquals() throws Exception {

        final RawLink link1 = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        final RawLink link2 = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        testAtomspaceTransaction(as ->
                Assert.assertEquals(as.get(link1).getId(), as.get(link2).getId()));
    }

    @Test
    public void testHashcode() throws Exception {

        final RawLink link1 = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        final RawLink link2 = new RawLink("Link",
                new RawNode("Node", "value1"),
                new RawNode("Node", "value2")
        );

        Assert.assertEquals(link1.hashCode(), link2.hashCode());

        testAtomspaceTransaction(as ->
                Assert.assertEquals(as.get(link1).hashCode(), as.get(link2).hashCode()));
    }

    @Test
    @Ignore
    public void testToString() throws Exception {

        final RawLink link = new RawLink("Link",
                new RawNode("Node1", "value1"),
                new RawNode("Node2", "value2")
        );

        Assert.assertEquals(
                "Link(Node1('value1'),Node2('value2'))",
                link.toString());

        testAtomspaceTransaction(as ->
                Assert.assertEquals(
                        "Link(Node1('value1'),Node2('value2'))",
                        as.get(link).toString()));
    }

    @Test
    public void testZeroArity() throws Exception {

        final RawLink link = new RawLink("Link");

        Assert.assertEquals(0, link.getArity());
        testAtomspaceTransaction(as -> {
            Assert.assertEquals(as.get("Link"), as.get(link));
        });
    }
}
