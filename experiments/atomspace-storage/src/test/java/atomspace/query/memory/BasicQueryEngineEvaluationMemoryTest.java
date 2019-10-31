package atomspace.query.memory;

import atomspace.query.AbstractBasicQueryEngineEvaluationTest;
import atomspace.query.AbstractBasicQueryEngineTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.memory.ASMemoryTestUtils;

public class BasicQueryEngineEvaluationMemoryTest extends AbstractBasicQueryEngineEvaluationTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASMemoryTestUtils.getTestStorage();
    }

}
