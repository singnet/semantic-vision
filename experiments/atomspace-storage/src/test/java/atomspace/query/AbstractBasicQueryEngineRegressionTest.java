package atomspace.query;

import atomspace.query.basic.ASBasicQueryEngine;
import atomspace.storage.ASAbstractTest;
import atomspace.storage.ASAtom;
import org.junit.Test;

public abstract class AbstractBasicQueryEngineRegressionTest extends ASAbstractTest {

    @Test
    public void testMatchSubTreeWithDifferentSize() throws Exception {

        testAtomspaceTransaction(as -> {

            // Link1(
            //  Node1("value1"),
            //  List2(
            //      Node2("value2"),
            //      Node3("value2")))
            ASAtom atom =
                    as.get("Link1",
                            as.get("Node1", "Value1"),
                            as.get("Link2",
                                    as.get("Node2", "Value2"),
                                    as.get("Node3", "Value3")));

            // Link1(
            //  Node1("value1"),
            //  List2(
            //      VariableNode("$VARIABLE")))
            ASAtom query =
                    as.get("Link1",
                            as.get("Node1", "Value1"),
                            as.get("Link2",
                                    as.get("VariableNode", "$VARIABLE")));

            ASQueryEngine queryEngine = new ASBasicQueryEngine();
            queryEngine.match(as, query);
        });
    }
}
