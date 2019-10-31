package atomspace.performance.runner;

import atomspace.performance.PerformanceModel;

public interface ModelRunner {

    PerformanceModel getModel(int param);

    void init(PerformanceModel model, StorageWrapper wrapper) throws Exception;

    void run(PerformanceModel model, StorageWrapper wrapper) throws Exception;
}
