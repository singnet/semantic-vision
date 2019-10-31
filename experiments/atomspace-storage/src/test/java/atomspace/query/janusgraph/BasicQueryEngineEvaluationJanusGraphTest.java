package atomspace.query.janusgraph;

import atomspace.query.AbstractBasicQueryEngineEvaluationTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.janusgraph.ASJanusGraphTestUtils;

public class BasicQueryEngineEvaluationJanusGraphTest extends AbstractBasicQueryEngineEvaluationTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }

}
