package atomspace.query.janusgraph;

import atomspace.query.AbstractBasicQueryEngineTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.janusgraph.ASJanusGraphTestUtils;

public class BasicQueryEngineJanusGraphTest extends AbstractBasicQueryEngineTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }

}
