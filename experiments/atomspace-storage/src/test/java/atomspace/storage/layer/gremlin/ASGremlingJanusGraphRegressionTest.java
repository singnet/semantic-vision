package atomspace.storage.layer.gremlin;

import atomspace.storage.ASAbstractRegressionTest;
import atomspace.storage.AtomspaceStorage;
import org.junit.Ignore;
import org.junit.Test;

public class ASGremlingJanusGraphRegressionTest extends ASAbstractRegressionTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return AtomspaceGremlinStorageHelper.getInMemoryJanusGraph(true, true);
    }

    /**
     * The top level link has the same type 'Link1' as its child link.
     * See the issue discussion on:
     * https://groups.google.com/forum/#!topic/janusgraph-users/x8Dh_V6Ue84
     */
    @Ignore
    @Test
    @Override
    public void testTopLevelLinkWithTheSameType() throws Exception {
        super.testTopLevelLinkWithTheSameType();
    }
}
