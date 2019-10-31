package atomspace.query.relationaldb;

import atomspace.query.AbstractBasicQueryEngineRepeatedVariablesTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.neo4j.ASNeo4jTestUtils;

public class BasicQueryEngineRepeatedVariablesRelationalDBTest extends AbstractBasicQueryEngineRepeatedVariablesTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }

}
