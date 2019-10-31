package atomspace.storage.memory;

import atomspace.storage.ASAbstractHelperTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.ASTransaction;
import atomspace.storage.util.AtomspaceStorageHelper;

public class ASMemoryHelperTest extends ASAbstractHelperTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }


    @Override
    public AtomspaceStorageHelper getHelper(AtomspaceStorage storage) {
        return ASMemoryTestUtils.getStorageHelper((AtomspaceMemoryStorage) storage);
    }
}
