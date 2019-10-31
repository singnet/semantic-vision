package atomspace.storage;

import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

public abstract class ASAbstractOutgoingListTest extends ASAbstractTest {

    @Test
    public void testEmptyOutgoingList() throws Exception {
        testAtomspaceTransaction(as -> {

            ASLink link = as.get("Link");
            ASOutgoingList outgoingList = link.getOutgoingList();

            Assert.assertEquals(0, outgoingList.getArity(as));
        });
    }

    @Test
    public void testOutgoingList1() throws Exception {
        testAtomspaceTransaction(as -> {

            ASAtom node = as.get("Node", "value");
            ASLink link = as.get("Link", node);
            ASOutgoingList outgoingList = link.getOutgoingList();

            Assert.assertEquals(1, outgoingList.getArity(as));
            Assert.assertEquals(node, outgoingList.getAtom(as,0));
        });
    }

    @Test
    public void testOutgoingList2() throws Exception {
        testAtomspaceTransaction(as -> {

            ASAtom node1 = as.get("Node1", "value1");
            ASAtom node2 = as.get("Node2", "value2");
            ASLink link = as.get("Link", node1, node2);
            ASOutgoingList outgoingList = link.getOutgoingList();

            Assert.assertEquals(2, outgoingList.getArity(as));
            Assert.assertEquals(node1, outgoingList.getAtom(as,0));
            Assert.assertEquals(node2, outgoingList.getAtom(as,1));
        });
    }

    @Test
    public void testOutgoingListN() throws Exception {
        testAtomspaceTransaction(as -> {

            final int max_values = 100;
            int n = new Random().nextInt(max_values);
            ASAtom[] atoms = new ASAtom[n];

            for (int i = 0; i < n; i++) {
                atoms[i] = as.get("Node", Integer.toString(i));
            }
            ASLink link = as.get("Link", atoms);
            ASOutgoingList outgoingList = link.getOutgoingList();

            Assert.assertEquals(n, outgoingList.getArity(as));
            for (int i = 0; i < n; i++) {
                Assert.assertEquals(atoms[i], outgoingList.getAtom(as,i));
            }
        });
    }
}
