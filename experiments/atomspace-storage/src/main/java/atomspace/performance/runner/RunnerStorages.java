package atomspace.performance.runner;

import atomspace.storage.layer.gremlin.AtomspaceGremlinStorage;
import atomspace.storage.layer.gremlin.AtomspaceGremlinStorageHelper;
import atomspace.storage.ASTransaction;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.janusgraph.AtomspaceJanusGraphStorage;
import atomspace.storage.janusgraph.AtomspaceJanusGraphStorageHelper;
import atomspace.storage.memory.AtomspaceMemoryStorage;
import atomspace.storage.memory.AtomspaceMemoryStorageHelper;
import atomspace.storage.neo4j.AtomspaceNeo4jStorage;
import atomspace.storage.neo4j.AtomspaceNeo4jStorageHelper;
import atomspace.storage.relationaldb.AtomspaceRelationalDBStorage;
import atomspace.storage.relationaldb.AtomspaceRelationalDBStorageHelper;
import atomspace.storage.util.AtomspaceStorageHelper;
import atomspace.storage.util.AtomspaceStorageUtils;
import org.janusgraph.core.JanusGraph;

public class RunnerStorages {

    public static AtomspaceMemoryStorage getMemoryStorage() {
        return new AtomspaceMemoryStorage();
    }

    public static AtomspaceRelationalDBStorage getRelationalDBStorage() {
        String dir = getTestStorageDir("relationaldb");
        AtomspaceStorageUtils.removeDirectory(dir);
        return AtomspaceRelationalDBStorageHelper.getInMemoryStorage(dir);
    }

    public static AtomspaceNeo4jStorage getNeo4jStorage() {
        String dir = getTestStorageDir("neo4j");
        AtomspaceStorageUtils.removeDirectory(dir);
        return new AtomspaceNeo4jStorage(dir);
    }

    private static String getTestStorageDir(String name) {
        return String.format("/tmp/atomspace-storage/performance/%s", name);
    }

    public static class DefaultStorageWrapper implements StorageWrapper {

        final String prefix;
        final int order;
        final String name;
        final AtomspaceStorage storage;
        final AtomspaceStorageHelper helper;


        public DefaultStorageWrapper(String prefix,
                                     int order,
                                     String name,
                                     AtomspaceStorage storage,
                                     AtomspaceStorageHelper helper) {
            this.prefix = prefix;
            this.order = order;
            this.name = name;
            this.storage = storage;
            this.helper = helper;
        }

        @Override
        public String getName() {
            return String.format("%s%d%s", prefix, order, name);
        }

        @Override
        public AtomspaceStorage getStorage() {
            return storage;
        }

        @Override
        public void printStatistics() throws Exception {
            try (ASTransaction tx = storage.getTx()) {
                helper.printStatistics(tx, "memory");
            }
        }

        @Override
        public void clean() throws Exception {
            try (ASTransaction tx = storage.getTx()) {
                helper.reset(tx);
                tx.commit();
            }
        }
    }

    public static StorageWrapper getMemoryStorageWrapper(String prefix) {
        AtomspaceMemoryStorage storage = getMemoryStorage();
        AtomspaceMemoryStorageHelper helper = new AtomspaceMemoryStorageHelper(storage);
        return new DefaultStorageWrapper(prefix, 1, "Memory", storage, helper);
    }

    public static StorageWrapper getRelationalDBStorageWrapper(String prefix) {
        AtomspaceRelationalDBStorage storage = getRelationalDBStorage();
        AtomspaceRelationalDBStorageHelper helper = new AtomspaceRelationalDBStorageHelper(storage);
        return new DefaultStorageWrapper(prefix, 2, "RelationalDB", storage, helper);
    }

    public static StorageWrapper getNeo4jStorageWrapper(String prefix) {
        AtomspaceNeo4jStorage storage = getNeo4jStorage();
        AtomspaceNeo4jStorageHelper helper = new AtomspaceNeo4jStorageHelper(storage);
        return new DefaultStorageWrapper(prefix, 3, "Neo4j", storage, helper);
    }

    public static StorageWrapper getJanusGraphStorageWrapper(String prefix, boolean useCustomIds) {
        JanusGraph graph = AtomspaceJanusGraphStorageHelper.getInMemoryJanusGraph(useCustomIds);
        AtomspaceJanusGraphStorage storage = new AtomspaceJanusGraphStorage(graph, useCustomIds);
        AtomspaceJanusGraphStorageHelper helper = new AtomspaceJanusGraphStorageHelper(storage);
        return new DefaultStorageWrapper(prefix, 4, "JanusGraph", storage, helper);
    }

    public static StorageWrapper getGremlingJanusGraphStorageWrapper(String prefix, boolean useCustomIds, boolean oneRequest) {
        AtomspaceGremlinStorage storage = AtomspaceGremlinStorageHelper.getInMemoryJanusGraph(useCustomIds, oneRequest);
        AtomspaceGremlinStorageHelper helper = new AtomspaceGremlinStorageHelper();
        int order = oneRequest ? 6 : 5;
        String label = oneRequest ? "GremlinOneRequest" : "GremlinMultipleRequests";
        return new DefaultStorageWrapper(prefix, order, label, storage, helper);
    }

    public static StorageWrapper getGremlingRemoteStorageWrapper(String prefix, String host, int port, boolean useCustomIds, boolean oneRequest) {
        AtomspaceGremlinStorage storage = AtomspaceGremlinStorageHelper.getRemoteJanusGraph(host, port, useCustomIds, oneRequest);
        AtomspaceGremlinStorageHelper helper = new AtomspaceGremlinStorageHelper();
        int order = oneRequest ? 6 : 5;
        String label = oneRequest ? "GremlinOneRequest" : "GremlinMultipleRequests";
        return new DefaultStorageWrapper(prefix, order, label, storage, helper);
    }
}
