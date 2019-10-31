package atomspace.storage;

import atomspace.ASTestUtils;
import atomspace.storage.util.AtomspaceStorageHelper;
import org.junit.Assert;
import org.junit.Test;

public abstract class ASAbstractHelperTest extends ASAbstractTest {

    public abstract AtomspaceStorageHelper getHelper(AtomspaceStorage storage);

    @Test
    public void testResetNode() throws Exception {
        testAtomspaceStorage(storage -> {

            try (ASTransaction as = storage.getTx()) {

                Assert.assertEquals(0, ASTestUtils.count(as.getAtoms()));

                as.get("Node", "value");
                Assert.assertEquals(1, ASTestUtils.count(as.getAtoms()));

                AtomspaceStorageHelper helper = getHelper(storage);
                helper.reset(as);
                Assert.assertEquals(0, ASTestUtils.count(as.getAtoms()));

                as.get("Node", "value");
                Assert.assertEquals(1, ASTestUtils.count(as.getAtoms()));
            }
        });
    }

    @Test
    public void testResetLink() throws Exception {
        testAtomspaceStorage(storage -> {

            try (ASTransaction as = storage.getTx()) {

                Assert.assertEquals(0, ASTestUtils.count(as.getAtoms()));

                as.get("Link",
                        as.get("Node", "value1"),
                        as.get("Node", "value2"));

                Assert.assertEquals(3, ASTestUtils.count(as.getAtoms()));

                AtomspaceStorageHelper helper = getHelper(storage);
                helper.reset(as);
                Assert.assertEquals(0, ASTestUtils.count(as.getAtoms()));

                as.get("Link",
                        as.get("Node", "value1"),
                        as.get("Node", "value2"));
                Assert.assertEquals(3, ASTestUtils.count(as.getAtoms()));
            }
        });
    }
}
