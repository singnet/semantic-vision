package atomspace.storage.memory;

import atomspace.storage.ASAbstractIncomingSetTest;
import atomspace.storage.AtomspaceStorage;

public class ASMemoryIncomingSetTest extends ASAbstractIncomingSetTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }
}
