package atomspace.query.neo4j;

import atomspace.query.AbstractBasicQueryEngineRepeatedVariablesTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.neo4j.ASNeo4jTestUtils;

public class BasicQueryEngineRepeatedVariablesNeo4jTest extends AbstractBasicQueryEngineRepeatedVariablesTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }

}
