package atomspace.query;

import atomspace.ASTestUtils;
import atomspace.ASTestUtils.TestQueryResult;
import atomspace.query.basic.ASBasicQueryEngine;
import atomspace.storage.ASAbstractTest;
import atomspace.storage.ASAtom;
import org.junit.Test;

import static atomspace.ASTestUtils.KeyWithValue;

public abstract class AbstractBasicQueryEngineEvaluationTest extends ASAbstractTest {

    @Test
    public void test1() throws Exception {
        testAtomspaceTransaction(as -> {

            ASAtom link = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject"),
                            as.get("ConceptNode", "object")));

            ASAtom query = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject"),
                            as.get("VariableNode", "$OBJECT")));


            ASQueryEngine queryEngine = new ASBasicQueryEngine();

            ASTestUtils.assertQueryResultsEqual(queryEngine.match(as, query),
                    new TestQueryResult(link, new KeyWithValue("$OBJECT", as.get("ConceptNode", "object"))));
        });
    }

    @Test
    public void test2() throws Exception {
        testAtomspaceTransaction(as -> {

            ASAtom link1 = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject"),
                            as.get("ConceptNode", "object1")));

            ASAtom link2 = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject"),
                            as.get("ConceptNode", "object2")));

            ASAtom query = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject"),
                            as.get("VariableNode", "$OBJECT")));

            ASQueryEngine queryEngine = new ASBasicQueryEngine();

            ASTestUtils.assertQueryResultsEqual(queryEngine.match(as, query),
                    new TestQueryResult(link1, new KeyWithValue("$OBJECT", as.get("ConceptNode", "object1"))),
                    new TestQueryResult(link2, new KeyWithValue("$OBJECT", as.get("ConceptNode", "object2"))));
        });
    }

    @Test
    public void test3() throws Exception {
        testAtomspaceTransaction(as -> {

            ASAtom link1 = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject"),
                            as.get("ConceptNode", "object1")));

            ASAtom link2 = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject"),
                            as.get("ConceptNode", "object2")));

            ASAtom link3 = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject3"),
                            as.get("ConceptNode", "object3")));

            ASAtom link4 = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate1"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject"),
                            as.get("ConceptNode", "object")));

            ASAtom query = as.get("EvaluationLink",
                    as.get("PredicateNode", "predicate"),
                    as.get("ListLink",
                            as.get("ConceptNode", "subject"),
                            as.get("VariableNode", "$OBJECT")));

            ASQueryEngine queryEngine = new ASBasicQueryEngine();

            ASTestUtils.assertQueryResultsEqual(queryEngine.match(as, query),
                    new TestQueryResult(link1, new KeyWithValue("$OBJECT", as.get("ConceptNode", "object1"))),
                    new TestQueryResult(link2, new KeyWithValue("$OBJECT", as.get("ConceptNode", "object2"))));
        });
    }
}
