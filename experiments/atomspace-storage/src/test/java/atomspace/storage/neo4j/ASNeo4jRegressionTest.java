package atomspace.storage.neo4j;

import atomspace.storage.ASAbstractRegressionTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.memory.ASMemoryTestUtils;

public class ASNeo4jRegressionTest extends ASAbstractRegressionTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }
}
