package atomspace.storage.janusgraph;

import atomspace.storage.ASAbstractNodeTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.memory.ASMemoryTestUtils;
import org.junit.Assert;
import org.junit.Test;

public class ASJanusGraphNodeTest extends ASAbstractNodeTest {

    @Override
    public AtomspaceStorage getStorage() {
        return ASJanusGraphTestUtils.getTestStorage();
    }
}
