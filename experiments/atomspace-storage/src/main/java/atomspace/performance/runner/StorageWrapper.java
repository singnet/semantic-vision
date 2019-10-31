package atomspace.performance.runner;

import atomspace.storage.AtomspaceStorage;

public interface StorageWrapper {

    String getName();

    AtomspaceStorage getStorage();

    void clean() throws Exception;

    void printStatistics() throws Exception;

    default void close() throws Exception {
        getStorage().close();
    }
}
