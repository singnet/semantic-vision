package atomspace.performance;

public interface PerformanceModelFactory {

    PerformanceModel getModel(int atoms, int queries);
}
