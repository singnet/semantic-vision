package atomspace.storage.neo4j;

import atomspace.storage.ASAbstractIncomingSetTest;
import atomspace.storage.AtomspaceStorage;

public class ASNeo4jIncomingSetTest extends ASAbstractIncomingSetTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }
}
