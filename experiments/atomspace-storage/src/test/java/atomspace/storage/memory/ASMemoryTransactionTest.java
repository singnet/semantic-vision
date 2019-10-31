package atomspace.storage.memory;

import atomspace.storage.ASAbstractTransactionTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.janusgraph.ASJanusGraphTestUtils;

public class ASMemoryTransactionTest extends ASAbstractTransactionTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }
}
