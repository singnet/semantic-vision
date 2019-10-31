package atomspace.storage.janusgraph;

import atomspace.storage.ASAbstractRegressionTest;
import atomspace.storage.AtomspaceStorage;

public class ASJanusGraphRegressionTest extends ASAbstractRegressionTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }
}
