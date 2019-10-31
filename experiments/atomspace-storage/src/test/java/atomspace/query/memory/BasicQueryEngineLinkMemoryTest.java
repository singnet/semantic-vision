package atomspace.query.memory;

import atomspace.query.AbstractBasicQueryEngineLinkTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.memory.ASMemoryTestUtils;

public class BasicQueryEngineLinkMemoryTest extends AbstractBasicQueryEngineLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }

}
