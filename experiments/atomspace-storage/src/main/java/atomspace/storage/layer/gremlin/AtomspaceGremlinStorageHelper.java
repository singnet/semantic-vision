package atomspace.storage.layer.gremlin;

import atomspace.storage.layer.gremlin.AtomspaceGremlinStorage.TransactionWithSource;
import atomspace.storage.ASTransaction;
import atomspace.storage.janusgraph.AtomspaceJanusGraphStorageHelper;
import atomspace.storage.janusgraph.JanusGraphUtils;
import atomspace.storage.util.AtomspaceStorageHelper;
import org.apache.commons.configuration.BaseConfiguration;
import org.apache.commons.configuration.Configuration;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.janusgraph.core.JanusGraph;
import org.janusgraph.core.JanusGraphFactory;
import org.janusgraph.graphdb.database.StandardJanusGraph;
import org.janusgraph.graphdb.idmanagement.IDManager;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicLong;

import static atomspace.storage.layer.gremlin.ASAbstractGremlinTransaction.*;

public class AtomspaceGremlinStorageHelper implements AtomspaceStorageHelper {

    @Override
    public void dump(ASTransaction tx) {
        ((ASAbstractGremlinTransaction) tx).dump();
    }

    @Override
    public void reset(ASTransaction tx) {
        ((ASAbstractGremlinTransaction) tx).reset();
    }

    @Override
    public void printStatistics(ASTransaction tx, String msg) {
        ((ASAbstractGremlinTransaction) tx).printStatistics(msg);
    }

    public static AtomspaceGremlinStorage getRemoteJanusGraph(String host, int port, boolean useCustomIds, boolean oneRequest) {
        GremlinRemoteStorage storage = new GremlinRemoteStorage(host, port, useCustomIds, oneRequest);
        return new AtomspaceGremlinStorage(storage);
    }

    public static AtomspaceGremlinStorage getInMemoryJanusGraph(boolean useCustomIds, boolean useOneRequest) {
        JanusGraph graph = AtomspaceJanusGraphStorageHelper.getInMemoryJanusGraph(useCustomIds);
        GremlinJanusGraphStorage storage = new GremlinJanusGraphStorage(graph, useCustomIds, useOneRequest);
        return new AtomspaceGremlinStorage(storage);
    }

    public static void dumpStorage(GraphTraversalSource g) {
        Iterator<Vertex> iter = g.V();
        while (iter.hasNext()) {
            Vertex v = iter.next();
            String kind = v.property(KIND).value().toString();
            String type = v.property(TYPE).value().toString();
            Object id = v.id();
            if (LABEL_NODE.equals(kind)) {
                String value = v.property(VALUE).value().toString();
                System.out.printf("%s[%s]: %s(%s)%n", kind, id, type, value);
            } else {
                long[] ids = (long[]) v.property(IDS).value();
                System.out.printf("%s[%s]: %s(%s)%n", kind, id, type, Arrays.toString(ids));
            }
        }
    }

    public static class GremlinRemoteStorage implements AtomspaceGremlinStorage.Storage {

        private final GraphTraversalSource g;
        private final boolean oneRequestTransaction;
        private final boolean useCustomIds;
        private final AtomicLong currentId = new AtomicLong();


        public GremlinRemoteStorage(String host, int port, boolean useCustomIds, boolean oneRequestTransaction) {
            this.useCustomIds = useCustomIds;
            this.oneRequestTransaction = oneRequestTransaction;

            try {
                this.g = JanusGraphFactory
                        .open("inmemory")
                        .traversal()
                        .withRemote(getConfig(host, port));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public TransactionWithSource get() {
            return new TransactionWithSource(g.tx(), g);
        }

        @Override
        public void initGraph() {
        }

        @Override
        public boolean oneRequest() {
            return oneRequestTransaction;
        }

        @Override
        public boolean useCustomIds() {
            return useCustomIds;
        }

        @Override
        public long getNextId() {
            return currentId.incrementAndGet();
        }

        @Override
        public void close() throws IOException {
            try {
                g.close();
            } catch (Exception e) {
                throw new IOException(e);
            }
        }

        private Configuration getConfig(String host, int port) {
            Configuration config = new BaseConfiguration();
            config.setProperty("graph.graphname", "atomspace");
            config.setProperty("clusterConfiguration.hosts", host);
            config.setProperty("clusterConfiguration.port", port);
            config.setProperty(
                    "clusterConfiguration.serializer.className",
                    "org.apache.tinkerpop.gremlin.driver.ser.GryoMessageSerializerV1d0");
            config.setProperty(
                    "serializer.config.ioRegistries",
                    "org.janusgraph.graphdb.tinkerpop.JanusGraphIoRegistry");
            config.setProperty(
                    "gremlin.remote.remoteConnectionClass",
                    "org.apache.tinkerpop.gremlin.driver.remote.DriverRemoteConnection");
            config.setProperty("gremlin.remote.driver.sourceName", "g");
            return config;
        }
    }

    public static class GremlinJanusGraphStorage implements AtomspaceGremlinStorage.Storage {

        private final JanusGraph graph;
        private final IDManager idManager;
        private final boolean useCustomIds;
        private final boolean useOneRequest;
        private final AtomicLong currentId = new AtomicLong();

        public GremlinJanusGraphStorage(JanusGraph graph, boolean useCustomIds, boolean useOneRequest) {
            this.graph = graph;
            this.useCustomIds = useCustomIds;
            this.useOneRequest = useOneRequest;
            this.idManager = ((StandardJanusGraph) graph).getIDManager();
        }

        @Override
        public TransactionWithSource get() {
            GraphTraversalSource g = graph.newTransaction().traversal();
            return new TransactionWithSource(g.tx(), g);
        }

        @Override
        public void initGraph() {
            JanusGraphUtils.makeIndices(graph);
        }

        @Override
        public boolean oneRequest() {
            return useOneRequest;
        }

        @Override
        public boolean useCustomIds() {
            return useCustomIds;
        }

        @Override
        public long getNextId() {
            return JanusGraphUtils.getNextId(idManager, currentId);
        }

        @Override
        public void close() {
            graph.close();
        }
    }
}
