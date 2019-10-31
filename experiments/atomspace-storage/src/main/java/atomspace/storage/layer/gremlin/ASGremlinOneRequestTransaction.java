package atomspace.storage.layer.gremlin;

import atomspace.storage.*;
import atomspace.storage.base.ASBaseLink;
import atomspace.storage.base.ASBaseNode;
import org.apache.tinkerpop.gremlin.process.traversal.Traverser;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversal;
import org.apache.tinkerpop.gremlin.structure.T;
import org.apache.tinkerpop.gremlin.structure.Vertex;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;

import static atomspace.storage.util.AtomspaceStorageUtils.getKey;
import static org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__.*;

public class ASGremlinOneRequestTransaction extends ASAbstractGremlinTransaction {

    public ASGremlinOneRequestTransaction(AtomspaceGremlinStorage.Storage storage) {
        super(storage);
    }

    @Override
    public ASLink get(RawLink link) {

        Vertex v = g
                .inject("nothing")
                .union(gremlinGetOrCreateLink(link))
                .next();

        return new ASBaseLink(id(v), link.getType(), link.getArity());
    }

    @Override
    public ASNode get(String type, String value) {

        Vertex v = g
                .inject("nothing")
                .union(gremlinGetOrCreateNode(type, value))
                .next();

        return new ASBaseNode(id(v), type, value);
    }

    @Override
    public ASLink get(String type, ASAtom... atoms) {
        long[] ids = getIds(atoms);

        GraphTraversal<Object, Vertex> addVertex = addV(LABEL_LINK)
                .property(KIND, LABEL_LINK)
                .property(TYPE, type)
                .property(IDS, ids);

        if (useCustomIds) {
            addVertex = addVertex.property(T.id, storage.getNextId());
        }

        int arity = atoms.length;
        for (int i = 0; i < arity; i++) {
            String key = getKey(type, arity, i);
            long id = atoms[i].getId();
            addVertex = addVertex.addE(key).to(g.V(id)).outV();
        }

        Vertex v = g
                .V()
                .hasLabel(LABEL_LINK)
                .has(TYPE, type)
                .has(IDS, ids)
                .fold()
                .coalesce(unfold(), addVertex)
                .next();

        return new ASBaseLink(id(v), type, atoms);
    }

    private GraphTraversal<Object, Vertex> gremlinGetOrCreateAtom(RawAtom atom) {
        if (atom instanceof RawNode) {
            return gremlinGetOrCreateNode((RawNode) atom);
        } else if (atom instanceof RawLink) {
            return gremlinGetOrCreateLink((RawLink) atom);
        } else {
            String msg = String.format("Unknown RawAtom class: %s", atom.getClass());
            throw new RuntimeException(msg);
        }
    }

    private GraphTraversal<Object, Vertex> gremlinGetOrCreateNode(RawNode node) {
        return gremlinGetOrCreateNode(node.getType(), node.getValue());
    }

    private GraphTraversal<Object, Vertex> gremlinGetOrCreateNode(String type, String value) {

        GraphTraversal<Object, Vertex> addVertex = addV(LABEL_NODE)
                .property(KIND, LABEL_NODE)
                .property(TYPE, type)
                .property(VALUE, value);

        if (useCustomIds) {
            addVertex = addVertex.property(T.id, storage.getNextId());
        }

        return V()
                .hasLabel(LABEL_NODE)
                .has(TYPE, type)
                .has(VALUE, value)
                .fold()
                .coalesce(unfold(), addVertex);
    }

    private GraphTraversal<Object, Vertex>[] gremlinGetOrCreateAtoms(RawLink link) {
        int arity = link.getArity();
        GraphTraversal<Object, Vertex>[] addAtoms = new GraphTraversal[arity];

        for (int i = 0; i < arity; i++) {
            addAtoms[i] = gremlinGetOrCreateAtom(link.getAtom(i));
        }
        return addAtoms;
    }

    private GraphTraversal<Object, Vertex> gremlinGetOrCreateLink(RawLink link) {

        Function<Traverser<Object>, Iterator<long[]>> mapIds = t -> {
            ArrayList arrayList = (ArrayList) t.get();
            long[] ids = new long[arrayList.size()];
            for (int i = 0; i < arrayList.size(); i++) {
                ids[i] = (long) arrayList.get(i);
            }
            List<long[]> list = new ArrayList<>(1);
            list.add(ids);
            return list.iterator();
        };

        GraphTraversal<Object, Vertex> addVertex = union(gremlinGetOrCreateAtoms(link))
                .id()
                .fold()
                .as("ids")
                .addV(LABEL_LINK)
                .property(KIND, LABEL_LINK)
                .property(TYPE, link.getType())
                .property(IDS, select("ids").flatMap(mapIds))
                .property(T.id, storage.getNextId());

        return union(gremlinGetOrCreateAtoms(link))
                .id()
                .fold()
                .as("ids")
                .V()
                .hasLabel(LABEL_LINK)
                .has(TYPE, link.getType())
                .has(IDS, select("ids").flatMap(mapIds))
                .fold()
                .coalesce(unfold(), addVertex);
    }
}
