package atomspace.query.neo4j;

import atomspace.query.AbstractBasicQueryEngineTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.neo4j.ASNeo4jTestUtils;

public class BasicQueryEngineNeo4jTest extends AbstractBasicQueryEngineTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }

}
