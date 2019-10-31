package atomspace.query.basic;

import atomspace.query.ASQueryEngine;
import atomspace.storage.*;

import java.util.*;
import java.util.function.Function;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ASBasicQueryEngine implements ASQueryEngine {

    private static final int MAX_COST = Integer.MAX_VALUE;
    private static Logger LOG = LoggerFactory.getLogger(ASBasicQueryEngine.class);

    public static final String TYPE_NODE_VARIABLE = "VariableNode";

    private static final int ROOT_DIRECTION = -1;
    private static final int UNDEFINED_DIRECTION = -2;

    @Override
    public <T> Iterator<T> match(ASTransaction tx, ASAtom query, Function<ASQueryResult, T> mapper) {

        LOG.trace("query {}", query);

        QueryTreeNode root = new QueryTreeNode(tx, null, query, ROOT_DIRECTION);
        QueryTreeNode startNode = findStartNode(tx, root);

        // Only support queries which contains at least one non variable node
        if (!startNode.isLeaf() || startNode.isVariable) {
            LOG.warn("skip query that contains only variables {}", query);
            return Collections.emptyIterator();
        }

        LOG.trace("start node {}", startNode.atom);

        Queue<QueryMatcherNode> queries = new ArrayDeque<>();
        queries.add(new QueryMatcherNode(UNDEFINED_DIRECTION, startNode, startNode.atom, new HashMap<>()));

        List<T> results = new LinkedList<>();

        while (!queries.isEmpty()) {
            QueryMatcherNode match = queries.poll();
            LOG.trace("node to match {}", match.rightAtom);

            // match subtree
            if (!matchSubTree(tx, match)) {
                continue;
            }

            // match root
            if (match.leftTreeNode.isRoot()) {
                LOG.trace("node accepted {}", match.rightAtom);
                results.add(mapper.apply(new ASBasicQueryResult(match.rightAtom, match.variables)));
                continue;
            }

            // match parent
            if (match.direction == ROOT_DIRECTION) {
                continue;
            }

            ASAtom rightAtom = match.rightAtom;

            ASIncomingSet incomingSet = rightAtom.getIncomingSet();

            QueryTreeNode parent = match.leftTreeNode.parent;
            String parentType = parent.atom.getType();
            int parentSize = parent.size;
            int parentPosition = match.leftTreeNode.parentPosition;

            Iterator<ASLink> iter = incomingSet.getIncomingSet(tx, parentType, parentSize, parentPosition);

            while (iter.hasNext()) {
                ASLink link = iter.next();
                QueryMatcherNode parentMatch = new QueryMatcherNode(parentPosition, parent, link, match.copyVariables());
                queries.add(parentMatch);
            }
        }

        return results.iterator();
    }

    QueryTreeNode findStartNode(ASTransaction tx, QueryTreeNode root) {
        NodeWithCost startNodeWithCost = findStartNode(tx, root, null, -1, -1);
        return startNodeWithCost.node;
    }

    NodeWithCost findStartNode(ASTransaction tx, QueryTreeNode node, String parentType, int parentSize, int position) {

        if (node.isVariable) {
            return new NodeWithCost(node, MAX_COST);
        }

        if (node.isLeaf()) {
            int cost = getCost(tx, node.atom, parentType, parentSize, position);
            return new NodeWithCost(node, cost);
        }

        QueryTreeNode currentNode = node;
        String type = node.atom.getType();
        int size = node.size;
        int currentCost = MAX_COST;
        QueryTreeNode[] children = node.children;

        for (int i = 0; i < children.length; i++) {
            NodeWithCost child = findStartNode(tx, children[i], type, size, i);

            if (child.cost <= currentCost) {
                currentNode = child.node;
                currentCost = child.cost;
            }
        }

        return new NodeWithCost(currentNode, currentCost);
    }

    boolean matchSubTree(ASTransaction tx, QueryMatcherNode match) {

        ASAtom rightAtom = match.rightAtom;
        ASAtom leftAtom = match.leftTreeNode.atom;

        // match right variable
        if (isVariable(rightAtom.getType())) {
            return false;
        }

        // match left variable
        if (match.leftTreeNode.isVariable) {

            String name = ((ASNode) leftAtom).getValue();
            ASAtom value = match.variables.get(name);

            if (value == null) {
                match.variables.put(name, match.rightAtom);
                return true;
            }

            return value.equals(rightAtom);
        }

        // match node
        if (match.leftTreeNode.isLeaf()) {
            return leftAtom.equals(rightAtom);
        }

        // match link
        if (!leftAtom.getType().equals(rightAtom.getType())) {
            return false;
        }

        ASOutgoingList outgoingList = ((ASLink) rightAtom).getOutgoingList();
        QueryTreeNode[] children = match.leftTreeNode.children;
        int size = children.length;

        // match outgoing list size
        if (size != outgoingList.getArity(tx)) {
            return false;
        }

        for (int i = 0; i < size; i++) {

            // Already visited
            if (match.direction == i) {
                continue;
            }

            ASAtom child = outgoingList.getAtom(tx, i);
            QueryMatcherNode childMatch = new QueryMatcherNode(ROOT_DIRECTION, children[i], child, match.variables);
            if (!matchSubTree(tx, childMatch)) {
                return false;
            }
        }

        return true;
    }


    static boolean isVariable(String type) {
        return TYPE_NODE_VARIABLE.equals(type);
    }

    static int getCost(ASTransaction tx, ASAtom atom, String type, int size, int position) {
        ASIncomingSet incomingSet = atom.getIncomingSet();
        return incomingSet.getIncomingSetSize(tx, type, size, position);
    }


    static class ASBasicQueryResult implements ASQueryResult {
        final ASAtom atom;
        final Map<String, ASAtom> variables;

        public ASBasicQueryResult(ASAtom atom, Map<String, ASAtom> variables) {
            this.atom = atom;
            this.variables = variables;
        }

        @Override
        public ASAtom getAtom() {
            return atom;
        }

        @Override
        public Map<String, ASAtom> getVariables() {
            return variables;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj instanceof ASQueryResult) {
                ASQueryResult that = (ASQueryResult) obj;
                return Objects.equals(this.getAtom(), that.getAtom()) &&
                        Objects.equals(this.getVariables(), that.getVariables());

            }
            return false;
        }

        @Override
        public int hashCode() {
            return Objects.hash(atom, variables);
        }

        @Override
        public String toString() {
            return String.format("atom: %s, variables: %s", atom, variables);
        }

    }

    static class NodeWithCost {
        final QueryTreeNode node;
        final int cost;

        public NodeWithCost(QueryTreeNode node, int cost) {
            this.node = node;
            this.cost = cost;
        }
    }

    static class QueryMatcherNode {
        final int direction;
        final QueryTreeNode leftTreeNode;
        final ASAtom rightAtom;
        final HashMap<String, ASAtom> variables;

        public QueryMatcherNode(int direction, QueryTreeNode leftTreeNode, ASAtom rightAtom, HashMap<String, ASAtom> variables) {
            this.direction = direction;
            this.leftTreeNode = leftTreeNode;
            this.rightAtom = rightAtom;
            this.variables = variables;
        }

        public HashMap<String, ASAtom> copyVariables() {
            return (HashMap<String, ASAtom>) (variables.clone());
        }
    }

    static class QueryTreeNode {

        final ASAtom atom;
        final int parentPosition;
        final QueryTreeNode parent;
        final QueryTreeNode[] children;
        final int size;
        final boolean isVariable;

        private static final QueryTreeNode[] EMPTY_CHILDREN = new QueryTreeNode[0];

        public QueryTreeNode(ASTransaction tx, QueryTreeNode parent, ASAtom atom, int parentPosition) {
            this.parent = parent;
            this.atom = atom;
            this.parentPosition = parentPosition;

            if (atom instanceof ASNode) {
                this.isVariable = isVariable(atom.getType());
                this.children = EMPTY_CHILDREN;
                this.size = 0;
            } else {
                this.isVariable = false;
                ASLink link = (ASLink) atom;
                ASOutgoingList outgoingList = link.getOutgoingList();
                int n = outgoingList.getArity(tx);
                this.children = new QueryTreeNode[n];
                this.size = n;

                for (int i = 0; i < n; i++) {
                    this.children[i] = new QueryTreeNode(tx, this, outgoingList.getAtom(tx, i), i);
                }
            }
        }

        public boolean isRoot() {
            return parent == null;
        }

        public boolean isLeaf() {
            return size == 0;
        }

        @Override
        public String toString() {
            return String.format("QueryTreeNode for atom: %s", atom);
        }
    }
}
