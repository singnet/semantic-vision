package atomspace.query.neo4j;

import atomspace.query.AbstractBasicQueryEngineRegressionTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.neo4j.ASNeo4jTestUtils;

public class BasicQueryEngineNeo4jRegressionTest extends AbstractBasicQueryEngineRegressionTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }

}
