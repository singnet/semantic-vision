package atomspace.storage.janusgraph;

import atomspace.storage.ASAbstractHelperTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.ASTransaction;
import atomspace.storage.util.AtomspaceStorageHelper;

public class ASJanusGraphHelperTest extends ASAbstractHelperTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }

    @Override
    public AtomspaceStorageHelper getHelper(AtomspaceStorage storage) {
        return ASJanusGraphTestUtils.getStorageHelper(storage);
    }
}
