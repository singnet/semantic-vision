package atomspace.storage.relationaldb;

import atomspace.storage.ASAbstractIncomingSetTest;
import atomspace.storage.AtomspaceStorage;

public class ASRelationalDBIncomingSetTest extends ASAbstractIncomingSetTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASRelationalDBTestUtils.getTestStorage();
    }
}
