package atomspace.storage.layer.gremlin;

import atomspace.storage.ASAtom;
import atomspace.storage.ASLink;
import atomspace.storage.ASNode;
import atomspace.storage.base.ASBaseLink;
import atomspace.storage.base.ASBaseNode;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversal;
import org.apache.tinkerpop.gremlin.structure.T;
import org.apache.tinkerpop.gremlin.structure.Vertex;

import static atomspace.storage.util.AtomspaceStorageUtils.getKey;

public class ASGremlinMultipleRequestTransaction extends ASAbstractGremlinTransaction {

    public ASGremlinMultipleRequestTransaction(AtomspaceGremlinStorage.Storage storage) {
        super(storage);
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

            int arity = atoms.length;
            for (int i = 0; i < arity; i++) {
                String key = getKey(type, arity, i);
                long id = atoms[i].getId();
                addVertex = addVertex.addE(key).to(g.V(id)).outV();
            }

            vertex = addVertex.next();
        }

        return new ASBaseLink(id(vertex), type, atoms);
    }
}
