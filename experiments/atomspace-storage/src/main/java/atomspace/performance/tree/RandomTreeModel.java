package atomspace.performance.tree;

import atomspace.performance.PerformanceModelConfiguration;
import atomspace.performance.PerformanceModelParameters;
import atomspace.query.ASQueryEngine;
import atomspace.query.ASQueryEngine.ASQueryResult;
import atomspace.query.basic.ASBasicQueryEngine;
import atomspace.storage.*;
import atomspace.storage.memory.AtomspaceMemoryStorage;
import atomspace.performance.PerformanceModel;

import java.io.IOException;
import java.util.*;
import java.util.function.Predicate;

public class RandomTreeModel implements PerformanceModel {

    private static final String VARIABLE_TYPE = ASBasicQueryEngine.TYPE_NODE_VARIABLE;

    public final PerformanceModelConfiguration config;
    public final PerformanceModelParameters params;
    public final RandomTreeModelParameters treeParams;
    public final Random random;

    private final List<NodeWithQuery> statements = new ArrayList<>();
    private final List<NodeWithQuery> queries = new ArrayList<>();

    public RandomTreeModel(PerformanceModelConfiguration config,
                           PerformanceModelParameters params,
                           RandomTreeModelParameters treeParams) {
        this.config = config;
        this.params = params;
        this.treeParams = treeParams;
        this.random = new Random(config.randomSeed + 2);

        this.init();
    }

    @Override
    public void createAtoms(AtomspaceStorage atomspace) throws IOException {

        ASTransaction tx = atomspace.getTx();

        int iterations = 0;
        int commits = 0;

        boolean useRawAtoms = params.useRawAtoms;

        for (NodeWithQuery nodeWithQuery : statements) {

            if (useRawAtoms) {
                tx.get(nodeWithQuery.rawAtom);
            } else {
                createAtom(tx, nodeWithQuery.node);
            }

            if (iterations++ >= params.iterationsBeforeCommit) {
                tx.commit();
                iterations = 0;
                commits++;
                tx = atomspace.getTx();
            }
        }

        tx.commit();
        commits++;
        //System.out.printf("commits: %d%n", commits);
    }

    private RawAtom createRawAtom(RandomNode node) {

        int n = node.children.length;

        if (n == 0) {
            return new RawNode(node.type, node.value);
        }

        RawAtom[] atoms = new RawAtom[n];
        for (int i = 0; i < n; i++) {
            atoms[i] = createRawAtom(node.children[i]);
        }

        return new RawLink(node.type, atoms);
    }

    private ASAtom createAtom(ASTransaction tx, RandomNode node) {

        int n = node.children.length;

        if (n == 0) {
            return tx.get(node.type, node.value);
        }

        ASAtom[] atoms = new ASAtom[n];
        for (int i = 0; i < n; i++) {
            atoms[i] = createAtom(tx, node.children[i]);
        }

        return tx.get(node.type, atoms);
    }

    @Override
    public void queryAtoms(AtomspaceStorage atomspace, ASQueryEngine queryEngine) throws Exception {

        try (ASTransaction tx = atomspace.getTx()) {
            for (NodeWithQuery pair : queries) {
                ASAtom query = createAtom(tx, pair.query);
                Iterator<ASQueryResult> results = queryEngine.match(tx, query);

                if (config.checkQueries) {

                    Set<ASAtom> atoms = new HashSet<>();
                    while (results.hasNext()) {
                        ASQueryResult result = results.next();
                        atoms.add(result.getAtom());
                    }

                    ASAtom atom = createAtom(tx, pair.node);

                    if (!atoms.contains(atom)) {
                        String msg = String.format("Atom: %s with query: %s was not found in results: %s%n",
                                query, atom, atoms);
                        throw new RuntimeException(msg);
                    }
                }
            }
        }
    }

    private final void init() {

        boolean useRawAtoms = params.useRawAtoms;
        loop:
        for (int i = 0; i < params.statements; i++) {

            RandomNode node = getNode(treeParams.maxWidth, treeParams.maxDepth - 1);
            while (true) {

                QueryItem queryItem = getQuery(node, treeParams.maxVariables);

                if (validQuery(queryItem.node)) {
                    RawAtom rawAtom = useRawAtoms ? createRawAtom(node) : null;
                    statements.add(new NodeWithQuery(node, queryItem.node, rawAtom));
                    continue loop;
                }
                node = getNode(treeParams.maxWidth, treeParams.maxDepth - 1);
            }
        }

        for (int i = 0; i < params.queries; i++) {
            int index = random.nextInt(statements.size());
            queries.add(statements.get(index));
        }
    }

    private RandomNode getNode(int width, int depth) {

        if (depth == 0) {
            return new RandomNode(getNodeType(), getValue());
        }

        int currentWidth = random.nextInt(width) + 1;
        int currentDepth = random.nextInt(depth) + 1;


        RandomNode[] children = new RandomNode[currentWidth];

        for (int i = 0; i < currentWidth; i++) {
            children[i] = getNode(currentWidth, currentDepth - 1);
        }

        return new RandomNode(getLinkType(), children);
    }

    private QueryItem getQuery(RandomNode node, int variables) {
        int n = node.children.length;

        if (n == 0) {

            if (convertNodeToVariable(variables)) {
                String variableName = String.format("$%s_%s", node.type.toUpperCase(), node.value.toUpperCase());
                return new QueryItem(new RandomNode(VARIABLE_TYPE, variableName), variables - 1);
            } else {
                return new QueryItem(node, variables);
            }
        }

        RandomNode[] children = new RandomNode[n];
        int vars = variables;
        for (int i = 0; i < n; i++) {
            QueryItem queryItem = getQuery(node.children[i], vars);
            children[i] = queryItem.node;
            vars = queryItem.variables;
        }

        return new QueryItem(new RandomNode(node.type, children), vars);
    }

    private String getNodeType() {
        return String.format("Node%d", random.nextInt(config.nodeTypes));
    }

    private String getLinkType() {
        return String.format("Link%d", random.nextInt(config.nodeTypes));
    }

    private String getValue() {
        return String.format("Value%d", random.nextInt(config.valuesPerType));
    }

    private boolean convertNodeToVariable(int variables) {
        return variables > 0 && random.nextInt(5) >= 2;
    }

    private static boolean validQuery(RandomNode node) {
        return hasVariable(node) && hasLeaf(node);
    }

    private static boolean hasVariable(RandomNode node) {
        return acceptRandomNode(node, n -> VARIABLE_TYPE.equals(n.type));
    }

    private static boolean hasLeaf(RandomNode node) {
        return acceptRandomNode(node, n -> !VARIABLE_TYPE.equals(n.type));
    }

    private static boolean acceptRandomNode(RandomNode node, Predicate<RandomNode> accept) {

        RandomNode[] children = node.children;

        if (children.length == 0) {
            return accept.test(node);
        }

        for (RandomNode child : node.children) {
            if (acceptRandomNode(child, accept)) {
                return true;
            }
        }
        return false;
    }

    public void dump() {
        System.out.printf("--- dump ---%n");
        System.out.printf("statements: %d%n", statements.size());
        for (NodeWithQuery pair : statements) {
            System.out.printf("%s%n", pair.node);
        }
        System.out.printf("queries: %d%n", queries.size());
        for (NodeWithQuery pair : queries) {
            System.out.printf("%s%n", pair.query);
        }
        System.out.printf("--- ---- ---%n");
    }

    @Override
    public String toString() {
        return String.format("Random tree model statements: %d, queries: %d", statements.size(), queries.size());
    }

    private static class QueryItem {
        final RandomNode node;
        final int variables;

        public QueryItem(RandomNode node, int variables) {
            this.node = node;
            this.variables = variables;
        }
    }

    private static class NodeWithQuery {
        final RandomNode node;
        final RandomNode query;
        final RawAtom rawAtom;

        public NodeWithQuery(RandomNode node, RandomNode query) {
            this(node, query, null);
        }

        public NodeWithQuery(RandomNode node, RandomNode query, RawAtom rawAtom) {
            this.node = node;
            this.query = query;
            this.rawAtom = rawAtom;
        }
    }

    private static class RandomNode {
        final String type;
        final String value;
        final RandomNode[] children;

        final static RandomNode[] EMPTY = {};

        public RandomNode(String type, String value) {
            this(type, value, EMPTY);
        }

        public RandomNode(String type, RandomNode... children) {
            this(type, "", children);
        }

        public RandomNode(String type, String value, RandomNode[] children) {
            this.type = type;
            this.value = value;
            this.children = children;
        }

        @Override
        public String toString() {
            StringBuilder builder = new StringBuilder();
            toString(builder, "");
            builder.append('\n');
            return builder.toString();
        }

        public void toString(StringBuilder builder, String indent) {
            builder
                    .append("\n")
                    .append(indent)
                    .append(type)
                    .append("(");
            if (children.length == 0) {
                builder
                        .append("'")
                        .append(value)
                        .append("'");
            } else {
                String nextIndent = indent + " ";
                for (RandomNode node : children) {
                    node.toString(builder, nextIndent);
                }
            }
            builder.append(")");
        }
    }

    public static void main(String[] args) throws Exception {

        PerformanceModelConfiguration config = new PerformanceModelConfiguration(3, 3, 3, true);
        PerformanceModelParameters params = new PerformanceModelParameters(5, 5);
        RandomTreeModelParameters treeParams = new RandomTreeModelParameters(3, 3, 2);
        RandomTreeModel model = new RandomTreeModel(config, params, treeParams);
        model.dump();

        try (AtomspaceStorage atomspace = new AtomspaceMemoryStorage();
             ASTransaction tx = atomspace.getTx()) {

            ASQueryEngine queryEngine = new ASBasicQueryEngine();
            model.createAtoms(atomspace);
            model.queryAtoms(atomspace, queryEngine);
        }
    }
}
