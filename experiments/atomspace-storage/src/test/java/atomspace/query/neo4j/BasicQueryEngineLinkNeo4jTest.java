package atomspace.query.neo4j;

import atomspace.query.AbstractBasicQueryEngineLinkTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.neo4j.ASNeo4jTestUtils;

public class BasicQueryEngineLinkNeo4jTest extends AbstractBasicQueryEngineLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASNeo4jTestUtils.getTestStorage();
    }

}
