package atomspace.query.janusgraph;

import atomspace.query.AbstractBasicQueryEngineRepeatedVariablesTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.janusgraph.ASJanusGraphTestUtils;

public class BasicQueryEngineRepeatedVariablesJanusGraphTest extends AbstractBasicQueryEngineRepeatedVariablesTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }

}
