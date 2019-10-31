package atomspace.performance.tree;

import atomspace.performance.PerformanceModel;
import atomspace.performance.PerformanceModelConfiguration;
import atomspace.performance.PerformanceModelParameters;
import atomspace.performance.runner.*;

import java.util.List;

import static atomspace.performance.runner.RunnerStorages.*;

public class RandomTreeModelGremlinCreateTest {

    public static void main(String[] args) throws Exception {

        String prefix = "create";
        boolean useCustomIds = true;

        // Run remote JanusGraph first.
        // For example using docker:
        // docker run --rm --name janusgraph-default \
        //    -e janusgraph.storage.berkeleyje.cache-percentage=80 \
        //    -e gremlinserver.threadPoolWorker=2 \
        //    -p 8182:8182 \
        //    janusgraph/janusgraph:latest


        //  String host = "localhost";
        //  int port = 8182;
        //  // Use these tests to test remote JanusGraph
        //  StorageWrapper[] wrappers = new StorageWrapper[]{
        //          getGremlingRemoteStorageWrapper(prefix, host, port, useCustomIds, false),
        //          getGremlingRemoteStorageWrapper(prefix, host, port, useCustomIds, true),
        //  };

        StorageWrapper[] wrappers = new StorageWrapper[]{
                getGremlingJanusGraphStorageWrapper(prefix, useCustomIds, false),
                getGremlingJanusGraphStorageWrapper(prefix, useCustomIds, true)
        };

        int[] statements = {100, 200, 300, 400, 500};
        ModelRunner runner = new RandomTreeCreateModelRunner(3, 3, 2, true);
        WarmupProperties warmup = new WarmupProperties(1, statements[2]);

        List<Measurement> results = RunnerUtils.measure(runner, wrappers, statements, warmup);

        for (Measurement result : results) {
            System.out.printf("result: %s%n", result);
        }

        RunnerUtils.showPlotter(results);
    }

    static class RandomTreeCreateModelRunner implements ModelRunner {

        final int randomTreeSize;
        final int maxTypes;
        final int maxVariables;
        final boolean useRawAtoms;

        public RandomTreeCreateModelRunner(int randomTreeSize, int maxTypes, int maxVariables, boolean useRawAtoms) {
            this.randomTreeSize = randomTreeSize;
            this.maxTypes = maxTypes;
            this.maxVariables = maxVariables;
            this.useRawAtoms = useRawAtoms;
        }


        @Override
        public PerformanceModel getModel(int statements) {
            PerformanceModelConfiguration config = new PerformanceModelConfiguration(maxTypes, maxTypes, maxTypes, true);
            PerformanceModelParameters params = new PerformanceModelParameters(statements, -1, 10, useRawAtoms);
            RandomTreeModelParameters treeParams = new RandomTreeModelParameters(randomTreeSize, randomTreeSize, maxVariables);
            return new RandomTreeModel(config, params, treeParams);
        }

        @Override
        public void init(PerformanceModel model, StorageWrapper wrapper) throws Exception {
            wrapper.clean();
        }

        @Override
        public void run(PerformanceModel model, StorageWrapper wrapper) throws Exception {
            model.createAtoms(wrapper.getStorage());
        }
    }
}

