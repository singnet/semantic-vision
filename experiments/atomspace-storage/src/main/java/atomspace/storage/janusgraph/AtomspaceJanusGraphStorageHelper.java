package atomspace.storage.janusgraph;

import atomspace.storage.ASTransaction;
import atomspace.storage.util.AtomspaceStorageHelper;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.janusgraph.core.JanusGraph;
import org.janusgraph.core.JanusGraphFactory;

import java.util.Iterator;

public class AtomspaceJanusGraphStorageHelper implements AtomspaceStorageHelper {

    private final AtomspaceJanusGraphStorage storage;

    public AtomspaceJanusGraphStorageHelper(AtomspaceJanusGraphStorage storage) {
        this.storage = storage;
    }

    @Override
    public void dump(ASTransaction tx) {
        ((ASJanusGraphTransaction) tx).dump();
    }

    @Override
    public void reset(ASTransaction tx) {
        ((ASJanusGraphTransaction) tx).reset();
    }

    @Override
    public void printStatistics(ASTransaction tx, String msg) {
        ((ASJanusGraphTransaction) tx).printStatistics(msg);
    }

    public static JanusGraph getInMemoryJanusGraph(boolean customIds) {

        JanusGraphFactory.Builder builder = JanusGraphFactory.build()
                .set("storage.backend", "inmemory")
                .set("ids.authority.wait-time", "5")
                .set("ids.renew-timeout", "50")
                .set("ids.block-size", "1000000000")
                .set("cluster.max-partitions", "2")
                .set("ids.renew-percentage", "0.2");

        if (customIds) {
            builder = builder.set("graph.set-vertex-id", "true");
        }

        return builder.open();
    }

    public static AtomspaceJanusGraphStorage getJanusGraphInMemoryStorage() {
        return new AtomspaceJanusGraphStorage(getInMemoryJanusGraph(true), true);
    }

    public static AtomspaceJanusGraphStorage getJanusGraphBerkeleyDBStorage(String storageDirectory) {
        JanusGraph graph = JanusGraphFactory.build()
                .set("storage.backend", "berkeleyje")
                .set("storage.directory", String.format("%s/graph", storageDirectory))
                .set("index.search.backend", "lucene")
                .set("index.search.directory", String.format("%s/index", storageDirectory))
                .set("graph.set-vertex-id", "true")
                //.set("ids.block-size", "100000")
                .set("ids.authority.wait-time", "5")
                //.set("ids.renew-timeout", "15")
                //.set("query.force-index", true)
                .open();
        return new AtomspaceJanusGraphStorage(graph, true);
    }
}
