package atomspace.storage.janusgraph;

import atomspace.storage.AtomspaceStorage;
import atomspace.storage.ASAbstractIncomingSetTest;

public class ASJanusGraphIncomingSetTest extends ASAbstractIncomingSetTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }
}
