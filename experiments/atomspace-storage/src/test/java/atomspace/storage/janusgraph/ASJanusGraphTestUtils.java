package atomspace.storage.janusgraph;

import atomspace.ASTestUtils;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.util.AtomspaceStorageHelper;

import java.io.IOException;

public class ASJanusGraphTestUtils {

    private static final String JANUSGRAPH_STORAGE_DIR;
    private static final AtomspaceJanusGraphStorage JANUS_GRAPHJ_STORAGE;

    static {
        try {
            JANUSGRAPH_STORAGE_DIR = ASTestUtils.getCleanNormalizedTempDir("atomspace-storage-junit-janusgraph");
            ASTestUtils.removeDirectory(JANUSGRAPH_STORAGE_DIR);
            JANUS_GRAPHJ_STORAGE = AtomspaceJanusGraphStorageHelper
                    .getJanusGraphBerkeleyDBStorage(JANUSGRAPH_STORAGE_DIR);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static AtomspaceJanusGraphStorage getTestStorage() {
        resetStorage();
        return JANUS_GRAPHJ_STORAGE;
    }

    public static AtomspaceJanusGraphStorageHelper getStorageHelper(AtomspaceStorage storage) {
        return new AtomspaceJanusGraphStorageHelper((AtomspaceJanusGraphStorage) storage);
    }

    private static void resetStorage() {
        try (ASJanusGraphTransaction tx = JANUS_GRAPHJ_STORAGE.getTx()) {
            AtomspaceStorageHelper helper = new AtomspaceJanusGraphStorageHelper(JANUS_GRAPHJ_STORAGE);
            helper.reset(tx);
            tx.commit();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
