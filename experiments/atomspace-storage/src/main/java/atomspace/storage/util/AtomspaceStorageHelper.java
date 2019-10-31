package atomspace.storage.util;

import atomspace.storage.ASTransaction;

public interface AtomspaceStorageHelper {

    void dump(ASTransaction tx);

    void reset(ASTransaction tx);

    void printStatistics(ASTransaction tx, String msg);
}
