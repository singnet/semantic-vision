package atomspace.storage.memory;

import atomspace.storage.*;

import java.util.HashMap;
import java.util.Map;

public class AtomspaceMemoryStorage implements AtomspaceStorage {


    static long UNIQUE_ID = 0;
    final Map<String, ASNode> nodesInverseIndex = new HashMap<>();
    final Map<String, ASLink> linksInverseIndex = new HashMap<>();

    private final ASMemoryTransaction tx = new ASMemoryTransaction(this);

    @Override
    public ASMemoryTransaction getTx() {
        return tx;
    }

    @Override
    public void close() {
    }

    long nextId() {
        return UNIQUE_ID++;
    }

    void reset() {
        nodesInverseIndex.clear();
        linksInverseIndex.clear();
        UNIQUE_ID = 0;
    }

    void printStatics(String msg) {
        int nodes = nodesInverseIndex.size();
        long links = linksInverseIndex.size();
        System.out.printf("%s nodes: %s, links: %s%n", msg, nodes, links);
    }
}
