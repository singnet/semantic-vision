package atomspace.storage.janusgraph;

import atomspace.storage.ASAbstractOutgoingListTest;
import atomspace.storage.AtomspaceStorage;

public class ASJanusGraphOutgoingListTest extends ASAbstractOutgoingListTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }
}
