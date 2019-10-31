package atomspace.performance.tree;

import atomspace.performance.PerformanceModel;
import atomspace.performance.PerformanceModelConfiguration;
import atomspace.performance.PerformanceModelParameters;
import atomspace.performance.runner.*;
import atomspace.query.ASQueryEngine;
import atomspace.query.basic.ASBasicQueryEngine;

import java.util.List;

import static atomspace.performance.runner.RunnerStorages.*;

public class RandomTreeModelQueryTest {

    public static void main(String[] args) throws Exception {

        String prefix = "query";

        StorageWrapper[] wrappers = new StorageWrapper[]{
                getMemoryStorageWrapper(prefix),
                getRelationalDBStorageWrapper(prefix),
                getNeo4jStorageWrapper(prefix),
                getJanusGraphStorageWrapper(prefix, true),
        };

        int[] queries = {100, 200, 300, 400, 500};
        ModelRunner runner = new RandomTreeQueryModelRunner(3, 3, 3, 10, false);
        WarmupProperties warmup = new WarmupProperties(1, queries[2]);

        List<Measurement> results = RunnerUtils.measure(runner, wrappers, queries, warmup);

        for (Measurement result : results) {
            System.out.printf("result: %s%n", result);
        }

        RunnerUtils.showPlotter(results);
    }

    static class RandomTreeQueryModelRunner implements ModelRunner {

        final int randomTreeSize;
        final int maxTypes;
        final int maxVariables;
        final int statements;
        public final boolean useRawAtoms;

        final ASQueryEngine queryEngine = new ASBasicQueryEngine();

        public RandomTreeQueryModelRunner(int randomTreeSize, int maxTypes, int maxVariables, int statements, boolean useRawAtoms) {
            this.randomTreeSize = randomTreeSize;
            this.maxTypes = maxTypes;
            this.maxVariables = maxVariables;
            this.statements = statements;
            this.useRawAtoms = useRawAtoms;
        }


        @Override
        public PerformanceModel getModel(int queries) {
            PerformanceModelConfiguration config = new PerformanceModelConfiguration(maxTypes, maxTypes, maxTypes, true);
            PerformanceModelParameters params = new PerformanceModelParameters(statements, queries, 10, false);
            RandomTreeModelParameters treeParams = new RandomTreeModelParameters(randomTreeSize, randomTreeSize, maxVariables);
            return new RandomTreeModel(config, params, treeParams);
        }

        @Override
        public void init(PerformanceModel model, StorageWrapper wrapper) throws Exception {
            wrapper.clean();
            model.createAtoms(wrapper.getStorage());
        }

        @Override
        public void run(PerformanceModel model, StorageWrapper wrapper) throws Exception {
            model.queryAtoms(wrapper.getStorage(), queryEngine);
        }
    }
}

