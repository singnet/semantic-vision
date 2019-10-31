package atomspace.performance.runner;

public class WarmupProperties {

    public final int iterations;
    public final int value;

    public WarmupProperties(int iterations, int value) {
        this.iterations = iterations;
        this.value = value;
    }
}
