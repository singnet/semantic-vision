package atomspace.storage.relationaldb;

import atomspace.storage.ASAbstractLinkTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.neo4j.ASNeo4jTestUtils;
import org.junit.Test;

public class ASRelationalDBLinkTest extends ASAbstractLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASRelationalDBTestUtils.getTestStorage();
    }
}
