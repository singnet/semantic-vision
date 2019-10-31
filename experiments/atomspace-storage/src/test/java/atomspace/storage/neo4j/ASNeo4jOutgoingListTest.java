package atomspace.storage.neo4j;

import atomspace.storage.ASAbstractOutgoingListTest;
import atomspace.storage.AtomspaceStorage;

public class ASNeo4jOutgoingListTest extends ASAbstractOutgoingListTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }
}
