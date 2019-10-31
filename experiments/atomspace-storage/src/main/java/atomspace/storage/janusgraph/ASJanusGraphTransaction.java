package atomspace.storage.janusgraph;

import atomspace.storage.ASAtom;
import atomspace.storage.ASLink;
import atomspace.storage.ASNode;
import atomspace.storage.ASTransaction;
import atomspace.storage.base.ASBaseLink;
import atomspace.storage.base.ASBaseNode;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversal;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Direction;
import org.apache.tinkerpop.gremlin.structure.Edge;
import org.apache.tinkerpop.gremlin.structure.T;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.janusgraph.core.JanusGraph;
import org.janusgraph.core.JanusGraphTransaction;
import org.janusgraph.core.JanusGraphVertex;
import org.janusgraph.core.Transaction;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static atomspace.storage.util.AtomspaceStorageUtils.count;
import static atomspace.storage.util.AtomspaceStorageUtils.getKey;

public class ASJanusGraphTransaction implements ASTransaction {

    // "type" is a reserved property name in JanusGraph
    static final String KIND = "as_kind";
    static final String TYPE = "as_type";
    static final String VALUE = "as_value";
    static final String IDS = "as_ids";

    static final String LABEL_NODE = "Node";
    static final String LABEL_LINK = "Link";

    final AtomspaceJanusGraphStorage storage;
    final JanusGraph graph;
    final JanusGraphTransaction tx;
    final GraphTraversalSource g;
    final boolean useCustomIds;

    public ASJanusGraphTransaction(AtomspaceJanusGraphStorage storage) {
        this.storage = storage;
        this.graph = storage.graph;
        this.useCustomIds = storage.useCustomIds;
        this.tx = graph.newTransaction();
        this.g = tx.traversal();
    }

    @Override
    public ASNode get(String type, String value) {

        GraphTraversal<Vertex, Vertex> iter = g
                .V()
                .hasLabel(LABEL_NODE)
                .has(TYPE, type)
                .has(VALUE, value);

        Vertex vertex = null;
        if (iter.hasNext()) {
            vertex = iter.next();
        }

        if (vertex == null) {
            GraphTraversal<Vertex, Vertex> addVertex = g
                    .addV(LABEL_NODE)
                    .property(KIND, LABEL_NODE)
                    .property(TYPE, type)
                    .property(VALUE, value);

            if (useCustomIds) {
                addVertex = addVertex.property(T.id, storage.getNextId());
            }

            vertex = addVertex.next();
        }

        return new ASBaseNode(id(vertex), type, value);
    }

    @Override
    public ASLink get(String type, ASAtom... atoms) {

        long[] ids = getIds(atoms);

        GraphTraversal<Vertex, Vertex> iter = g
                .V()
                .hasLabel(LABEL_LINK)
                .has(TYPE, type)
                .has(IDS, ids);

        Vertex vertex = null;
        if (iter.hasNext()) {
            vertex = iter.next();
        }

        if (vertex == null) {
            GraphTraversal<Vertex, Vertex> addVertex = g
                    .addV(LABEL_LINK)
                    .property(KIND, LABEL_LINK)
                    .property(TYPE, type)
                    .property(IDS, ids);

            if (useCustomIds) {
                addVertex = addVertex.property(T.id, storage.getNextId());
            }

            vertex = addVertex.next();

            // Update incoming set
            int arity = atoms.length;
            for (int i = 0; i < arity; i++) {
                String key = getKey(type, arity, i);
                Vertex childVertex = g.V(atoms[i].getId()).next();
                childVertex.addEdge(key, vertex);
            }
        }

        return new ASBaseLink(id(vertex), type, atoms);
    }

    @Override
    public ASAtom get(long id) {
        Vertex vertex = vertex(id);
        String kind = vertex.property(KIND).value().toString();
        String type = vertex.property(TYPE).value().toString();

        if (LABEL_NODE.equals(kind)) {
            String value = vertex.property(VALUE).value().toString();
            return new ASBaseNode(id, type, value);
        }

        if (LABEL_LINK.equals(kind)) {
            long[] ids = (long[]) vertex.property(IDS).value();
            return new ASBaseLink(id, type, ids);
        }

        String msg = String.format("Unknown kind: %s%n", kind);
        throw new RuntimeException(msg);
    }

    @Override
    public long[] getOutgoingListIds(long id) {
        return (long[]) vertex(id).property(IDS).value();
    }

    @Override
    public int getIncomingSetSize(long id, String type, int arity, int position) {
        // TBD: use the count store
        return count(getSet(id, type, arity, position));
    }

    @Override
    public Iterator<ASLink> getIncomingSet(long id, String type, int arity, int position) {
        List<ASLink> links = new ArrayList<>();
        Iterator<Edge> iter = getSet(id, type, arity, position);
        while (iter.hasNext()) {
            JanusGraphVertex parent = (JanusGraphVertex) iter.next().inVertex();
            links.add(new ASBaseLink(id(parent), type, arity));
        }

        return links.iterator();
    }

    private Iterator<Edge> getSet(long id, String type, int size, int position) {
        Vertex vertex = g.V(id).next();
        String key = getKey(type, size, position);
        return vertex.edges(Direction.OUT, key);
    }

    @Override
    public Iterator<ASAtom> getAtoms() {
        printStatistics("[get atoms]");

        List<ASAtom> atoms = new ArrayList<>();

        GraphTraversal<Vertex, Vertex> nodes = g.V().has(KIND, LABEL_NODE);

        while (nodes.hasNext()) {
            ASAtom node = get(id(nodes.next()));
            atoms.add(node);
        }

        GraphTraversal<Vertex, Vertex> links = g.V().has(KIND, LABEL_LINK);

        while (links.hasNext()) {
            ASAtom link = get(id(links.next()));
            atoms.add(link);
        }

        return atoms.iterator();
    }

    @Override
    public void commit() {
        tx.commit();
    }

    @Override
    public void close() {
        tx.close();
    }

    private static long id(Vertex v) {
        return (long) v.id();
    }

    private Vertex vertex(long id) {
        return g.V(id).next();
    }

    private static long[] getIds(ASAtom... atoms) {
        long[] ids = new long[atoms.length];

        for (int i = 0; i < atoms.length; i++) {
            ids[i] = atoms[i].getId();
        }
        return ids;
    }

    void reset() {
        g.E().drop().iterate();
        g.V().drop().iterate();
    }

    void printStatistics(String msg) {
        long nodes = g.V().has(KIND, LABEL_NODE).count().next();
        long links = g.V().has(KIND, LABEL_LINK).count().next();
        System.out.printf("%s nodes: %s, links: %s%n", msg, nodes, links);
    }

    void dump() {
        System.out.printf("--- Dump JanusGraph ---%n");
        Iterator<Vertex> iter = g.V();
        while (iter.hasNext()) {
            Vertex v = iter.next();
            String type = v.property(TYPE).value().toString();
            System.out.printf("type: %s%n", type);
        }
        System.out.printf("--- --------------- ---%n");
    }
}
