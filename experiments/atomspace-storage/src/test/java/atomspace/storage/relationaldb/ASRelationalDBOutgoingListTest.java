package atomspace.storage.relationaldb;

import atomspace.storage.ASAbstractOutgoingListTest;
import atomspace.storage.AtomspaceStorage;

public class ASRelationalDBOutgoingListTest extends ASAbstractOutgoingListTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASRelationalDBTestUtils.getTestStorage();
    }
}
