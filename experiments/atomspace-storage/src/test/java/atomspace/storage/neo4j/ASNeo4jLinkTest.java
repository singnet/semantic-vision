package atomspace.storage.neo4j;

import atomspace.storage.ASAbstractLinkTest;
import atomspace.storage.AtomspaceStorage;

public class ASNeo4jLinkTest extends ASAbstractLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }
}
