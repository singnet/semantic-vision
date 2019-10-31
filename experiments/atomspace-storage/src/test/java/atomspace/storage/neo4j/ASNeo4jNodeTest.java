package atomspace.storage.neo4j;

import atomspace.storage.ASAbstractNodeTest;
import atomspace.storage.AtomspaceStorage;

public class ASNeo4jNodeTest extends ASAbstractNodeTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }
}
