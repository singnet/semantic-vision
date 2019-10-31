package atomspace.query;

import atomspace.query.basic.ASBasicQueryEngine;
import atomspace.storage.ASAbstractTest;
import atomspace.storage.ASAtom;
import org.junit.Ignore;
import org.junit.Test;

import static atomspace.ASTestUtils.*;

public abstract class AbstractBasicQueryEngineTest extends ASAbstractTest {

    @Ignore
    @Test
    // Use Typed variables
    public void test0() throws Exception {
        testAtomspaceTransaction(as -> {

            ASAtom atom = as.get("Node", "value");
            ASAtom query = as.get("VariableNode", "$VALUE");

            ASQueryEngine queryEngine = new ASBasicQueryEngine();
            assertQueryResultsEqual(queryEngine.match(as, query),
                    new TestQueryResult(atom, new KeyWithValue("$VALUE", as.get("Node", "value"))));
        });
    }

    @Test
    public void test1() throws Exception {
        testAtomspaceTransaction(as -> {

            ASAtom atom = as.get("PredicateLink",
                    as.get("SubjectNode", "subject"),
                    as.get("ObjectNode", "object"));

            ASAtom query = as.get("PredicateLink",
                    as.get("SubjectNode", "subject"),
                    as.get("VariableNode", "$WHAT"));

            ASQueryEngine queryEngine = new ASBasicQueryEngine();

            assertQueryResultsEqual(queryEngine.match(as, query),
                    new TestQueryResult(atom, new KeyWithValue("$WHAT", as.get("ObjectNode", "object"))));
        });
    }

    @Test
    public void test2() throws Exception {
        testAtomspaceTransaction(as -> {

            ASAtom atom1 = as.get("PredicateLink",
                    as.get("SubjectNode", "subject"),
                    as.get("ObjectNode", "object1"));

            ASAtom atom2 = as.get("PredicateLink",
                    as.get("SubjectNode", "subject"),
                    as.get("ObjectNode", "object2"));

            ASAtom atom3 = as.get("PredicateLink",
                    as.get("SubjectNode", "subject3"),
                    as.get("ObjectNode", "object3"));

            ASAtom query = as.get("PredicateLink",
                    as.get("SubjectNode", "subject"),
                    as.get("VariableNode", "$WHAT"));

            ASQueryEngine queryEngine = new ASBasicQueryEngine();

            assertQueryResultsEqual(queryEngine.match(as, query),
                    new TestQueryResult(atom2, new KeyWithValue("$WHAT", as.get("ObjectNode", "object2"))),
                    new TestQueryResult(atom1, new KeyWithValue("$WHAT", as.get("ObjectNode", "object1"))));
        });
    }
}
