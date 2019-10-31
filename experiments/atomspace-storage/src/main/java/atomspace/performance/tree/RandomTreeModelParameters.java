package atomspace.performance.tree;

public class RandomTreeModelParameters {
    public final int maxWidth;
    public final int maxDepth;
    public final int maxVariables;

    public RandomTreeModelParameters(int maxWidth, int maxDepth, int maxVariables) {
        this.maxWidth = maxWidth;
        this.maxDepth = maxDepth;
        this.maxVariables = maxVariables;
    }
}
