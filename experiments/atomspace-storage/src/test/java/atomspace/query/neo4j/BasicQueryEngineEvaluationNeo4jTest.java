package atomspace.query.neo4j;

import atomspace.query.AbstractBasicQueryEngineEvaluationTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.neo4j.ASNeo4jTestUtils;

public class BasicQueryEngineEvaluationNeo4jTest extends AbstractBasicQueryEngineEvaluationTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }

}
