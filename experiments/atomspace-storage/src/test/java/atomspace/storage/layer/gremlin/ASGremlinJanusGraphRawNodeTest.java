package atomspace.storage.layer.gremlin;

import atomspace.storage.ASAbstractRawNodeTest;
import atomspace.storage.AtomspaceStorage;

public class ASGremlinJanusGraphRawNodeTest extends ASAbstractRawNodeTest {

    @Override
    public AtomspaceStorage getStorage() {
        return AtomspaceGremlinStorageHelper.getInMemoryJanusGraph(true, true);
    }
}
