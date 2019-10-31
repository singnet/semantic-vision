package atomspace.storage.memory;

import atomspace.storage.ASTransaction;
import atomspace.storage.util.AtomspaceStorageHelper;

public class AtomspaceMemoryStorageHelper implements AtomspaceStorageHelper {


    private final AtomspaceMemoryStorage storage;

    public AtomspaceMemoryStorageHelper(AtomspaceMemoryStorage storage) {
        this.storage = storage;
    }

    @Override
    public void dump(ASTransaction tx) {
    }

    @Override
    public void reset(ASTransaction tx) {
        ((ASMemoryTransaction) tx).reset();
    }

    @Override
    public void printStatistics(ASTransaction tx, String msg) {
        storage.printStatics(msg);
    }
}
