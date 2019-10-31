package atomspace.performance.runner;

import java.util.*;

import atomspace.performance.PerformanceModel;
import atomspace.performance.tree.RandomTreeModel;

import static atomspace.performance.runner.ResultPlotter.PlotterProperties.toPointList;


public class RunnerUtils {

    public static List<Measurement> measure(ModelRunner runner, StorageWrapper[] wrappers, int[] xs, WarmupProperties warmup) throws Exception {

        Map<String, List<Long>> map = new HashMap<>();

        // warmup
        PerformanceModel model = runner.getModel(warmup.value);
        for (int i = 0; i < warmup.iterations; i++) {
            System.out.printf("Warmup%n");
            for (StorageWrapper wrapper : wrappers) {
                runner.init(model, wrapper);
                runner.run(model, wrapper);
                wrapper.printStatistics();
            }
        }

        // performance measurement
        for (int x : xs) {
            model = runner.getModel(x);
            System.out.printf("%nMeasure results x: %d, %s%n", x, model);
            for (StorageWrapper wrapper : wrappers) {
                runner.init(model, wrapper);
                long time = System.currentTimeMillis();
                runner.run(model, wrapper);
                long elapsedTime = System.currentTimeMillis() - time;
                wrapper.printStatistics();

                List<Long> ys = map.computeIfAbsent(wrapper.getName(), (key) -> new LinkedList<>());
                ys.add(elapsedTime);
            }
        }

        for (StorageWrapper wrapper : wrappers) {
            wrapper.close();
        }

        double[] xsd = Arrays.stream(xs).mapToDouble(x -> x).toArray();
        List<Measurement> measurements = new ArrayList<>(wrappers.length);

        for (Map.Entry<String, List<Long>> entry : map.entrySet()) {
            String name = entry.getKey();
            double[] ysd = entry.getValue().stream().mapToDouble(y -> y).toArray();
            Measurement measurement = new Measurement(name, xsd, ysd);
            measurements.add(measurement);
        }

        return measurements;
    }

    public static void showPlotter(List<Measurement> measurements) {
        Map<String, List<ResultPlotter.PointDouble>> map = new HashMap<>();
        for (Measurement measurement : measurements) {
            double[] xs = measurement.xs;
            double[] ys = measurement.ys;
            map.put(measurement.name, toPointList(xs, ys));
        }

        ResultPlotter.PlotterProperties properties = ResultPlotter.PlotterProperties.differentCharts(map);
        ResultPlotter.showMeasurements(properties);
    }
}
