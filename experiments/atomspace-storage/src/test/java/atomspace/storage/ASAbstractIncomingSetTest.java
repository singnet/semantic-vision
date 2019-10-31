package atomspace.storage;

import atomspace.ASTestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.Iterator;

public abstract class ASAbstractIncomingSetTest extends ASAbstractTest {

    // Node("value")
    @Test
    public void testEmptyIncomingSet() throws Exception {

        testAtomspaceTransaction(as -> {

            ASNode node = as.get("Node", "value");
            assertIncomingSet(as, node, "Node", 0, 0);
            assertIncomingSet(as, node, "Node", 1, 0);
            assertIncomingSet(as, node, "Node", 1, 1);
        });
    }

    // Link(Node("value"))
    @Test
    public void testIncomingSet1() throws Exception {

        testAtomspaceTransaction(as -> {

            ASNode node = as.get("Node", "value");
            ASLink link = as.get("Link", node);

            ASIncomingSet nodeIncomingSet = node.getIncomingSet();
            assertIncomingSet(as, nodeIncomingSet, "Node", 0, 0);
            assertIncomingSet(as, nodeIncomingSet, "Link", 1, 0, link);
            assertIncomingSet(as, nodeIncomingSet, "Link", 1, 1);
        });
    }


    // Link(Node("value"), Node("value"))
    @Test
    public void testIncomingSet11() throws Exception {

        testAtomspaceTransaction(as -> {

            ASNode node = as.get("Node", "value");
            ASLink link = as.get("Link", node, node);

            ASIncomingSet nodeIncomingSet = node.getIncomingSet();
            assertIncomingSet(as, nodeIncomingSet, "Link", 0, 0);
            assertIncomingSet(as, nodeIncomingSet, "Link", 0, 1);
            assertIncomingSet(as, nodeIncomingSet, "Link", 1, 0);
            assertIncomingSet(as, nodeIncomingSet, "Link", 2, 0, link);
            assertIncomingSet(as, nodeIncomingSet, "Link", 2, 1, link);
        });
    }


    // Link(Node1("value"), Node2("value"))
    @Test
    public void testIncomingSet12() throws Exception {

        testAtomspaceTransaction(as -> {

            ASNode node1 = as.get("Node1", "value");
            ASNode node2 = as.get("Node2", "value");
            ASLink link = as.get("Link", node1, node2);

            // node1
            ASIncomingSet nodeIncomingSet1 = node1.getIncomingSet();

            assertIncomingSet(as, nodeIncomingSet1, "Link", 2, 0, link);
            assertIncomingSet(as, nodeIncomingSet1, "Link", 2, 1);

            // node2
            ASIncomingSet nodeIncomingSet2 = node2.getIncomingSet();

            assertIncomingSet(as, nodeIncomingSet2, "Link", 2, 0);
            assertIncomingSet(as, nodeIncomingSet2, "Link", 2, 1, link);
        });
    }


    // Link1(Node("value")), Link2(Node("value"))
    @Test
    public void testIncomingSetLinks12() throws Exception {

        testAtomspaceTransaction(as -> {

            ASAtom node = as.get("Node", "value");
            ASLink link1 = as.get("Link1", node);
            ASLink link2 = as.get("Link2", node);

            ASIncomingSet incomingSet = node.getIncomingSet();

            assertIncomingSet(as, incomingSet, "Link1", 1, 0, link1);
            assertIncomingSet(as, incomingSet, "Link2", 1, 0, link2);
        });
    }

    // Link(SubjectNode("subject"), ObjectNode("object1"))
    // Link(SubjectNode("subject"), ObjectNode("object2"))
    @Test
    public void testIncomingSetCommonChild() throws Exception {

        testAtomspaceTransaction(as -> {

            ASLink link1 = as.get("Link",
                    as.get("SubjectNode", "subject"),
                    as.get("ObjectNode", "object1"));

            ASLink link2 = as.get("Link",
                    as.get("SubjectNode", "subject"),
                    as.get("ObjectNode", "object2"));

            ASIncomingSet incomingSet = as.get("SubjectNode", "subject").getIncomingSet();

            assertIncomingSet(as, incomingSet, "Link", 2, 0, link1, link2);
        });
    }

    private static void assertIncomingSet(ASTransaction tx,
                                          ASAtom atom,
                                          String type,
                                          int arity,
                                          int position,
                                          ASLink... links) {
        assertIncomingSet(tx, atom.getIncomingSet(), type, arity, position, links);
    }

    private static void assertIncomingSet(ASTransaction tx,
                                          ASIncomingSet incomingSet,
                                          String type,
                                          int arity,
                                          int position,
                                          ASLink... links) {
        Assert.assertEquals(links.length, incomingSet.getIncomingSetSize(tx, type, arity, position));
        Iterator<ASLink> iter = incomingSet.getIncomingSet(tx, type, arity, position);
        ASTestUtils.assertIteratorEquals(iter, links);
    }
}
