package atomspace.storage.layer.gremlin;

import atomspace.storage.ASTransaction;
import atomspace.storage.AtomspaceStorage;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Transaction;

import java.io.Closeable;

public class AtomspaceGremlinStorage implements AtomspaceStorage {

    final Storage storage;

    public AtomspaceGremlinStorage(Storage storage) {
        this.storage = storage;
        storage.initGraph();
    }

    @Override
    public ASTransaction getTx() {
        return  storage.oneRequest()
                ? new ASGremlinOneRequestTransaction(storage)
                : new ASGremlinMultipleRequestTransaction(storage);
    }

    @Override
    public void close() throws Exception {
        storage.close();
    }

    interface Storage extends Closeable {

        TransactionWithSource get();

        void initGraph();

        boolean oneRequest();

        boolean useCustomIds();

        long getNextId();
    }

    static class TransactionWithSource {

        private final Transaction tx;
        private final GraphTraversalSource g;

        public TransactionWithSource(Transaction tx, GraphTraversalSource g) {
            this.tx = tx;
            this.g = g;
        }

        public Transaction tx() {
            return tx;
        }

        public GraphTraversalSource traversal() {
            return g;
        }
    }
}
