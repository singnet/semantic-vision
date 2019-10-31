package atomspace.query.relationaldb;

import atomspace.query.AbstractBasicQueryEngineTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.relationaldb.ASRelationalDBTestUtils;

public class BasicQueryEngineRelationalDBTest extends AbstractBasicQueryEngineTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASRelationalDBTestUtils.getTestStorage();
    }
}
