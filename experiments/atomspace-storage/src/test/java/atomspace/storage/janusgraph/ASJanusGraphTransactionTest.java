package atomspace.storage.janusgraph;

import atomspace.storage.ASAbstractTransactionTest;
import atomspace.storage.AtomspaceStorage;

public class ASJanusGraphTransactionTest extends ASAbstractTransactionTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }
}
