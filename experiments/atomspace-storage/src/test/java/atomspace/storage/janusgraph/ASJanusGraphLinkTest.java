package atomspace.storage.janusgraph;

import atomspace.storage.ASAbstractLinkTest;
import atomspace.storage.AtomspaceStorage;

public class ASJanusGraphLinkTest extends ASAbstractLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }
}
