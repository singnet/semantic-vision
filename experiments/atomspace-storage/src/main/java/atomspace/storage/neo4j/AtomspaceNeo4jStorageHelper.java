package atomspace.storage.neo4j;

import atomspace.storage.ASTransaction;
import atomspace.storage.util.AtomspaceStorageHelper;

public class AtomspaceNeo4jStorageHelper implements AtomspaceStorageHelper {

    private final AtomspaceNeo4jStorage storage;

    public AtomspaceNeo4jStorageHelper(AtomspaceNeo4jStorage storage) {
        this.storage = storage;
    }

    @Override
    public void dump(ASTransaction tx) {
    }

    @Override
    public void reset(ASTransaction tx) {
        ((ASNeo4jTransaction) tx).reset();
    }

    @Override
    public void printStatistics(ASTransaction tx, String msg) {
        ((ASNeo4jTransaction) tx).printStatics(msg);
    }
}
