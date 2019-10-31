package atomspace.storage.layer.gremlin;

import atomspace.storage.ASAtom;
import atomspace.storage.ASLink;
import atomspace.storage.ASTransaction;
import atomspace.storage.base.ASBaseLink;
import atomspace.storage.base.ASBaseNode;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Transaction;
import org.apache.tinkerpop.gremlin.structure.Vertex;

import java.io.IOException;
import java.util.Iterator;

public abstract class ASAbstractGremlinTransaction implements ASTransaction {

    // "type" is a reserved property name in JanusGraph
    protected static final String KIND = "as_kind";
    protected static final String TYPE = "as_type";
    protected static final String VALUE = "as_value";
    protected static final String IDS = "as_ids";

    protected static final String LABEL_NODE = "Node";
    protected static final String LABEL_LINK = "Link";

    protected final AtomspaceGremlinStorage.Storage storage;
    protected final Transaction tx;
    protected final GraphTraversalSource g;
    protected final boolean useCustomIds;

    public ASAbstractGremlinTransaction(AtomspaceGremlinStorage.Storage storage) {
        this.storage = storage;
        AtomspaceGremlinStorage.TransactionWithSource pair = storage.get();
        this.tx = pair.tx();
        this.g = pair.traversal();
        this.useCustomIds = storage.useCustomIds();
    }

    @Override
    public ASAtom get(long id) {

        Vertex v = g.V(id).next();
        String kind = v.property(KIND).value().toString();
        String type = v.property(TYPE).value().toString();

        switch (kind) {
            case LABEL_NODE: {
                String value = v.property(VALUE).value().toString();
                return new ASBaseNode(id, type, value);
            }
            case LABEL_LINK: {
                long[] ids = (long[]) v.property(IDS).value();
                return new ASBaseLink(id, type, ids);
            }
            default: {
                String msg = String.format("Unknown atom kind: %s", kind);
                throw new RuntimeException(msg);
            }
        }
    }

    @Override
    public long[] getOutgoingListIds(long id) {
        Vertex v = g.V(id).next();
        return (long[]) v.property(IDS).value();
    }

    @Override
    public int getIncomingSetSize(long id, String type, int arity, int position) {
        return 0;
    }

    @Override
    public Iterator<ASLink> getIncomingSet(long id, String type, int arity, int position) {
        return null;
    }

    @Override
    public Iterator<ASAtom> getAtoms() {
        return null;
    }

    @Override
    public void commit() {
        tx.commit();
    }

    @Override
    public void close() throws IOException {
        tx.close();
    }

    protected void reset() {
        g.E().drop().iterate();
        g.V().drop().iterate();
    }

    protected void dump() {
        System.out.printf("--- Gremlin Storage Dump ---%n");
        AtomspaceGremlinStorageHelper.dumpStorage(g);
        System.out.printf("--- -------------------- ---%n");
    }

    protected void printStatistics(String msg) {
        long nodes = g.V().has(KIND, LABEL_NODE).count().next();
        long links = g.V().has(KIND, LABEL_LINK).count().next();
        System.out.printf("%s nodes: %s, links: %s%n", msg, nodes, links);
    }


    protected static long id(Vertex v) {
        return (long) v.id();
    }

    protected static long[] getIds(ASAtom... atoms) {
        long[] ids = new long[atoms.length];

        for (int i = 0; i < atoms.length; i++) {
            ids[i] = atoms[i].getId();
        }
        return ids;
    }
}
