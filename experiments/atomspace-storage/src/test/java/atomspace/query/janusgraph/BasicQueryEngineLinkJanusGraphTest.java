package atomspace.query.janusgraph;

import atomspace.query.AbstractBasicQueryEngineLinkTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.janusgraph.ASJanusGraphTestUtils;

public class BasicQueryEngineLinkJanusGraphTest extends AbstractBasicQueryEngineLinkTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }

}
