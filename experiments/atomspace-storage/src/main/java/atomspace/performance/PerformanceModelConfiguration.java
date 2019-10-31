package atomspace.performance;

public class PerformanceModelConfiguration {

    public final long randomSeed;
    public final int nodeTypes;
    public final int linkTypes;
    public final int valuesPerType;
    public final boolean checkQueries;

    public PerformanceModelConfiguration(int nodeTypes, int linkTypes, int valuesPerType, boolean checkQueries) {
        this(42, nodeTypes, linkTypes, valuesPerType, checkQueries);
    }

    public PerformanceModelConfiguration(long randomSeed, int nodeTypes, int linkTypes, int valuesPerType, boolean checkQueries) {
        this.randomSeed = randomSeed;
        this.nodeTypes = nodeTypes;
        this.linkTypes = linkTypes;
        this.valuesPerType = valuesPerType;
        this.checkQueries = checkQueries;
    }
}
