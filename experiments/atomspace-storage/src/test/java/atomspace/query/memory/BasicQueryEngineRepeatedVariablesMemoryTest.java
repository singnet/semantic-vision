package atomspace.query.memory;

import atomspace.query.AbstractBasicQueryEngineRepeatedVariablesTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.memory.ASMemoryTestUtils;

public class BasicQueryEngineRepeatedVariablesMemoryTest extends AbstractBasicQueryEngineRepeatedVariablesTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }

}
