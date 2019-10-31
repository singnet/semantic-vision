package atomspace.storage.layer.gremlin;

import atomspace.storage.ASAbstractLinkTest;
import atomspace.storage.AtomspaceStorage;

public class ASGremlingJanusGraphLinkTest extends ASAbstractLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return AtomspaceGremlinStorageHelper.getInMemoryJanusGraph(true, true);
    }
}
