package atomspace.performance;

import atomspace.query.ASQueryEngine;
import atomspace.storage.AtomspaceStorage;

public interface PerformanceModel {

    void createAtoms(AtomspaceStorage atomspace) throws Exception;

    void queryAtoms(AtomspaceStorage atomspace, ASQueryEngine queryEngine) throws Exception;
}