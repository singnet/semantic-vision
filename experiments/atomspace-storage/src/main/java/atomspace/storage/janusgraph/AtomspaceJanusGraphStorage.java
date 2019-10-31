package atomspace.storage.janusgraph;

import atomspace.storage.AtomspaceStorage;
import org.janusgraph.core.JanusGraph;
import org.janusgraph.graphdb.database.StandardJanusGraph;
import org.janusgraph.graphdb.idmanagement.IDManager;

import java.util.concurrent.atomic.AtomicLong;

public class AtomspaceJanusGraphStorage implements AtomspaceStorage {

    final JanusGraph graph;
    final IDManager idManager;
    final AtomicLong currentId = new AtomicLong();
    final boolean useCustomIds;

    public AtomspaceJanusGraphStorage(JanusGraph graph, boolean useCustomIds) {
        this.graph = graph;
        this.useCustomIds = useCustomIds;
        this.idManager = ((StandardJanusGraph) graph).getIDManager();
        JanusGraphUtils.makeIndices(graph);
    }

    @Override
    public ASJanusGraphTransaction getTx() {
        return new ASJanusGraphTransaction(this);
    }

    @Override
    public void close() throws Exception {
        graph.close();
    }

    long getNextId() {
        return JanusGraphUtils.getNextId(idManager, currentId);
    }
}
