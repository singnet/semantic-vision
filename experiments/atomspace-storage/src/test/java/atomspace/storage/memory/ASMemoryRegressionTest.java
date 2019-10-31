package atomspace.storage.memory;

import atomspace.storage.ASAbstractRegressionTest;
import atomspace.storage.AtomspaceStorage;

public class ASMemoryRegressionTest extends ASAbstractRegressionTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }
}
