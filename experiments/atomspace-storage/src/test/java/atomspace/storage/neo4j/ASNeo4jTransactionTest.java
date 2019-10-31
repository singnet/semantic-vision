package atomspace.storage.neo4j;

import atomspace.storage.ASAbstractTransactionTest;
import atomspace.storage.AtomspaceStorage;

public class ASNeo4jTransactionTest extends ASAbstractTransactionTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }
}
