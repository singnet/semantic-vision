package atomspace.query.relationaldb;

import atomspace.query.AbstractBasicQueryEngineLinkTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.relationaldb.ASRelationalDBTestUtils;

public class BasicQueryEngineLinkRelationalDBTest extends AbstractBasicQueryEngineLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASRelationalDBTestUtils.getTestStorage();
    }

}
