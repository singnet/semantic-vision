package atomspace.storage;

import org.junit.Assert;
import org.junit.Test;

import java.util.Iterator;

public abstract class ASAbstractTransactionTest extends ASAbstractTest {

    @Test
    public void testTransaction() throws Exception {

        AtomspaceStorage storage = getStorage();

        try (ASTransaction tx = storage.getTx()) {
            Iterator<ASAtom> iter = tx.getAtoms();
            Assert.assertFalse(iter.hasNext());
            tx.commit();
        }

        try (ASTransaction tx = storage.getTx()) {
            tx.get("Node", "value");
            Iterator<ASAtom> iter = tx.getAtoms();
            Assert.assertTrue(iter.hasNext());
            checkNode(iter.next(), "Node", "value");
            Assert.assertFalse(iter.hasNext());
            tx.commit();
        }

        try (ASTransaction tx = storage.getTx()) {
            Iterator<ASAtom> iter = tx.getAtoms();
            Assert.assertTrue(iter.hasNext());
            checkNode(iter.next(), "Node", "value");
            Assert.assertFalse(iter.hasNext());
            tx.commit();
        }
    }

    static void checkNode(ASAtom atom, String type, String value) {

        Assert.assertEquals(type, atom.getType());

        Assert.assertTrue("Atom is not a Node!", atom instanceof ASNode);
        ASNode node = (ASNode) atom;
        Assert.assertEquals(value, node.getValue());
    }
}
