package atomspace.query.janusgraph;

import atomspace.query.AbstractBasicQueryEngineRegressionTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.janusgraph.ASJanusGraphTestUtils;

public class BasicQueryEngineJanusGraphRegressionTest extends AbstractBasicQueryEngineRegressionTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }

}
