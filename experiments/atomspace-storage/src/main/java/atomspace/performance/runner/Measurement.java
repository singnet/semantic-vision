package atomspace.performance.runner;

public class Measurement {

    public final String name;
    public final double[] xs;
    public final double[] ys;

    public Measurement(String name, double[] xs, double[] ys) {
        this.name = name;
        this.xs = xs;
        this.ys = ys;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(name);
        for (int i = 0; i < xs.length; i++) {
            builder
                    .append(" (")
                    .append(xs[i])
                    .append(", ")
                    .append(ys[i])
                    .append(")");
        }
        return builder.toString();
    }
}
