package atomspace.query.relationaldb;

import atomspace.query.AbstractBasicQueryEngineEvaluationTest;
import atomspace.storage.AtomspaceStorage;
import atomspace.storage.relationaldb.ASRelationalDBTestUtils;

public class BasicQueryEngineEvaluationRelationalDBTest extends AbstractBasicQueryEngineEvaluationTest {

    @Override
    protected AtomspaceStorage getStorage() {
        return ASRelationalDBTestUtils.getTestStorage();
    }

}
