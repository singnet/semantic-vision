package atomspace.storage.relationaldb;

import atomspace.storage.ASAbstractNodeTest;
import atomspace.storage.AtomspaceStorage;
import org.junit.Assert;
import org.junit.Test;

public class ASRelationalDBNodeTest extends ASAbstractNodeTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASRelationalDBTestUtils.getTestStorage();
    }
}
