package atomspace.performance.runner;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import java.util.*;

public class ResultPlotter extends Application {

    // Platform.startup(...) method is not available in JDK 8.
    private static PlotterProperties PROPS;

    @Override
    public void start(Stage stage) {

        stage.setTitle(PROPS.title);

        List<LineChart<Number, Number>> lineCharts =
                getLineCharts(PROPS);
        HBox hbox = new HBox(lineCharts.toArray(new LineChart[]{}));

        Scene scene = new Scene(hbox, PROPS.width, PROPS.height);
        stage.setScene(scene);
        stage.show();
    }

    List<LineChart<Number, Number>> getLineCharts(PlotterProperties props) {

        List<LineChart<Number, Number>> charts = new ArrayList<>();

        Bounds bounds = getBounds(props.measurements);

        if (props.sameChart) {
            charts.add(getLineChart("", bounds, props));
        }

        Set<Map.Entry<String, List<PointDouble>>> entries = props.measurements.entrySet();
        TreeSet<Map.Entry<String, List<PointDouble>>> sortedEntries = new TreeSet<>(Comparator.comparing(Map.Entry::getKey));
        sortedEntries.addAll(entries);

        for (Map.Entry<String, List<PointDouble>> entry : sortedEntries) {
            String name = entry.getKey();
            String seriesName = props.sameChart ? name : "";
            List<PointDouble> values = entry.getValue();

            LineChart<Number, Number> lineChart = props.sameChart
                    ? charts.get(0)
                    : getLineChart(name, getBounds(values), props);

            lineChart.getData().addAll(getSeries(seriesName, values));

            if (!props.sameChart) {
                lineChart.setLegendVisible(false);
                charts.add(lineChart);
            }
        }

        return charts;
    }

    LineChart<Number, Number> getLineChart(String name, Bounds b, PlotterProperties props) {

        final NumberAxis xAxis = new NumberAxis(b.xMin, b.xMax, (b.xMax - b.xMin) / 8);
        final NumberAxis yAxis = new NumberAxis(b.yMin, b.yMax, (b.yMax - b.yMin) / 8);

        LabelWithTitle labelWithTitle = getLabelWithTitle(name);

        LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);
        yAxis.setLabel(String.format("Time(%s)", props.timeUnits));
        xAxis.setLabel(labelWithTitle.label);
        lineChart.setTitle(labelWithTitle.title);

        return lineChart;
    }


    static XYChart.Series getSeries(String name, List<PointDouble> values) {
        XYChart.Series series = new XYChart.Series();
        series.setName(name);

        for (PointDouble value : values) {
            double x = value.x;
            double y = value.y;
            series.getData().add(new XYChart.Data(x, y));
        }

        return series;
    }

    public static void showMeasurements(PlotterProperties props) {
        // Platform.startup(...) method is not available in JDK 8.
        PROPS = props;
        launch();
    }

    public static void main(String[] args) {

        Map<String, List<PointDouble>> measurements = new HashMap<>();

        List<PointDouble> list = new ArrayList<>();
        list.add(new PointDouble(2, 4));
        list.add(new PointDouble(4, 8));
        list.add(new PointDouble(6, 12));
        list.add(new PointDouble(8, 16));
        measurements.put("testA", list);

        list = new ArrayList<>();
        list.add(new PointDouble(2, 4));
        list.add(new PointDouble(4, 8));
        list.add(new PointDouble(6, 16));
        list.add(new PointDouble(8, 32));
        measurements.put("testB", list);

        list = new ArrayList<>();
        list.add(new PointDouble(2, 4));
        list.add(new PointDouble(4, 5));
        list.add(new PointDouble(6, 6));
        list.add(new PointDouble(8, 7));
        measurements.put("testC", list);

        list = new ArrayList<>();
        list.add(new PointDouble(2, 7));
        list.add(new PointDouble(4, 6));
        list.add(new PointDouble(6, 5));
        list.add(new PointDouble(8, 4));
        measurements.put("testD", list);

//        PlotterProperties props = PlotterProperties.sameChart(measurements);
        PlotterProperties props = PlotterProperties.differentCharts(measurements);

        showMeasurements(props);
    }

    Bounds getBounds(Map<String, List<PointDouble>> measurements) {
        Bounds bounds = new Bounds(Double.MAX_VALUE, Double.MIN_VALUE, Double.MAX_VALUE, Double.MIN_VALUE);

        for (List<PointDouble> values : measurements.values()) {
            bounds = bounds.union(getBounds(values));
        }

        return bounds;
    }

    Bounds getBounds(List<PointDouble> values) {
        double xMin = Double.MAX_VALUE;
        double xMax = Double.MIN_VALUE;

        double yMin = Double.MAX_VALUE;
        double yMax = Double.MIN_VALUE;

        for (PointDouble value : values) {

            double x = value.x;
            double y = value.y;

            if (x < xMin) xMin = x;
            if (x > xMax) xMax = x;
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
        }

        return new Bounds(xMin, xMax, yMin, yMax);
    }

    static LabelWithTitle getLabelWithTitle(String str) {

        for (int i = 0; i < str.length(); i++) {

            char c = str.charAt(i);

            if (Character.isUpperCase(str.charAt(i))) {
                String label = str.substring(0, i);
                String title = str.substring(i);
                return new LabelWithTitle(label, title);
            }

            if (Character.isDigit(str.charAt(i))) {
                String label = str.substring(0, i);
                String title = str.substring(i + 1);
                return new LabelWithTitle(label, title);
            }
        }

        return new LabelWithTitle(str, "");
    }

    static class LabelWithTitle {
        public final String label;
        public final String title;

        public LabelWithTitle(String label, String title) {
            this.label = label;
            this.title = title;
        }

        @Override
        public String toString() {
            return String.format("label: %s, title: %s", label, title);
        }
    }

    public static class PointDouble {
        final double x;
        final double y;

        public PointDouble(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }

    static class Bounds {
        final double xMin;
        final double xMax;

        final double yMin;
        final double yMax;

        public Bounds(double xMin, double xMax, double yMin, double yMax) {
            this.xMin = xMin;
            this.xMax = xMax;
            this.yMin = yMin;
            this.yMax = yMax;
        }

        Bounds union(Bounds that) {

            return new Bounds(
                    Math.min(this.xMin, that.xMin),
                    Math.max(this.xMax, that.xMax),
                    Math.min(this.yMin, that.yMin),
                    Math.max(this.yMax, that.yMax));
        }
    }

    public static class PlotterProperties {
        public int width = 700;
        public int height = 300;
        public String title = "";
        public String timeUnits = "ms";
        public boolean sameChart = true;
        public Map<String, List<PointDouble>> measurements;

        public static PlotterProperties sameChart(Map<String, List<PointDouble>> measurements) {
            PlotterProperties props = new PlotterProperties();
            props.sameChart = true;
            props.measurements = measurements;
            return props;
        }

        public static PlotterProperties differentCharts(Map<String, List<PointDouble>> measurements) {
            PlotterProperties props = new PlotterProperties();
            props.sameChart = false;
            props.measurements = measurements;
            return props;
        }

        public static List<PointDouble> toPointList(double[] xs, double[] ys) {
            List<PointDouble> points = new ArrayList<>(xs.length);

            for (int i = 0; i < xs.length; i++) {
                points.add(new PointDouble(xs[i], ys[i]));
            }
            return points;
        }
    }
}
