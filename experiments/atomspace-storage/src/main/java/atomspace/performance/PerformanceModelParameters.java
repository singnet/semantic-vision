package atomspace.performance;

public class PerformanceModelParameters {
    public final int statements;
    public final int queries;
    public final int iterationsBeforeCommit;
    public final boolean useRawAtoms;


    public PerformanceModelParameters(int statements, int queries) {
        this(statements, queries, 1, false);
    }

    public PerformanceModelParameters(int statements, int queries, int iterationsBeforeCommit, boolean useRawAtoms) {
        this.statements = statements;
        this.queries = queries;
        this.iterationsBeforeCommit = iterationsBeforeCommit;
        this.useRawAtoms = useRawAtoms;
    }
}
