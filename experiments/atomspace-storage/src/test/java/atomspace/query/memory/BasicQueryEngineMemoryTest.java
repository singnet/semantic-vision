package atomspace.query.memory;

import atomspace.query.AbstractBasicQueryEngineRegressionTest;
import atomspace.query.AbstractBasicQueryEngineTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.memory.ASMemoryTestUtils;

public class BasicQueryEngineMemoryTest extends AbstractBasicQueryEngineTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }

}
