package atomspace.storage.memory;

import atomspace.storage.ASAbstractOutgoingListTest;
import atomspace.storage.AtomspaceStorage;

public class ASMemoryOutgoingListTest extends ASAbstractOutgoingListTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }
}
