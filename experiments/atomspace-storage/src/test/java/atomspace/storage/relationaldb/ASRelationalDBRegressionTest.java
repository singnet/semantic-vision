package atomspace.storage.relationaldb;

import atomspace.storage.ASAbstractRegressionTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.neo4j.ASNeo4jTestUtils;

public class ASRelationalDBRegressionTest extends ASAbstractRegressionTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASRelationalDBTestUtils.getTestStorage();
    }
}
