package atomspace.storage;

import org.junit.Assert;
import org.junit.Test;

public abstract class ASAbstractNodeTest extends ASAbstractTest {

    @Test
    public void testNotNull() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertNotNull(as.get("Node", "value")));
    }

    @Test
    public void testEquals() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertEquals(
                        as.get("Node", "value"),
                        as.get("Node", "value")));
    }

    @Test
    public void testIdEquals() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertEquals(
                        as.get("Node", "value").getId(),
                        as.get("Node", "value").getId()));
    }

    @Test
    public void testHashcode() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertEquals(
                        as.get("Node", "value").hashCode(),
                        as.get("Node", "value").hashCode()));
    }

    @Test
    public void testToString() throws Exception {

        testAtomspaceTransaction(as ->
                Assert.assertEquals(
                        "Node('value')",
                        as.get("Node", "value").toString()));
    }
}
