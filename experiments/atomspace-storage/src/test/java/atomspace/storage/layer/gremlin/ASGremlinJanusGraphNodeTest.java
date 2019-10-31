package atomspace.storage.layer.gremlin;

import atomspace.storage.ASAbstractNodeTest;
import atomspace.storage.AtomspaceStorage;

public class ASGremlinJanusGraphNodeTest extends ASAbstractNodeTest {

    @Override
    public AtomspaceStorage getStorage() {
        return AtomspaceGremlinStorageHelper.getInMemoryJanusGraph(true, true);
    }
}
