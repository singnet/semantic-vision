package atomspace.storage.neo4j;

import atomspace.storage.AtomspaceStorage;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.factory.GraphDatabaseFactory;

import java.io.File;

public class AtomspaceNeo4jStorage implements AtomspaceStorage {

    final GraphDatabaseService graph;

    public AtomspaceNeo4jStorage(String storageDirectory) {
        this.graph = new GraphDatabaseFactory().newEmbeddedDatabase(new File(storageDirectory));
        makeIndices();
    }

    @Override
    public ASNeo4jTransaction getTx() {
        return new ASNeo4jTransaction(graph);
    }

    @Override
    public void close() {
        graph.shutdown();
    }

    private void makeIndices() {
        try (Transaction tx = graph.beginTx()) {

            graph
                    .schema()
                    .indexFor(Label.label("Node"))
                    .on("type")
                    .on("value")
                    .create();

            graph
                    .schema()
                    .indexFor(Label.label("Link"))
                    .on("type")
                    .on("ids")
                    .create();

            tx.success();
        }
    }
}
