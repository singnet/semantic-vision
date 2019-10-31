package atomspace.storage.relationaldb;

import atomspace.storage.ASAbstractTransactionTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.neo4j.ASNeo4jTestUtils;

public class ASRelationalDBTransactionTest extends ASAbstractTransactionTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASRelationalDBTestUtils.getTestStorage();
    }
}
