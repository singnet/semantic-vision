package org.opencog.vqa;

import java.util.List;
import java.util.stream.Collectors;

class RelexPredicate implements Comparable<RelexPredicate> {
    private final String name;
    private final List<RelexArgument> arguments;

    public RelexPredicate(String name, RelexArgument firstArg, RelexArgument secondArg) {
        this.name = name;
        this.arguments = List.of(firstArg, secondArg);
        firstArg.addRelation(this);
        secondArg.addRelation(this);
    }

    @Override
    public int compareTo(RelexPredicate other) {
        if (getNumberOfArgumentUsages() != other.getNumberOfArgumentUsages()) {
            return getNumberOfArgumentUsages() - other.getNumberOfArgumentUsages();
        }
        return name.compareTo(other.name);
    }

    private int getNumberOfArgumentUsages() {
        return arguments.stream().collect(Collectors.summingInt(RelexArgument::getNumberOfUsages));
    }

    public String toFormula() {
        return name + "(" + arguments.stream().map(fn -> fn.getVariableName()).collect(Collectors.joining(", "))
                + ")";
    }

    public String toShortFormula() {
        return name + "()";
    }

    public String toAtomeeseFormula() {
        if (name.equals("_predadj")) {
            String object = arguments.get(0).getVariableName();
            String firstPredicate = arguments.get(0).getName();
            String secondPredicate = arguments.get(1).getName();
            return String.format("(AndLink " +
                    "(InheritanceNode (VariableNode \"$%1$s\") (ConceptNode \"BoundingBox\"))" +
                    "(GroundedPredicateNode \"py: %2$s\" (VariableNode \"$%1$s\"))" +
                    "(GroundedPredicateNode \"py: %3$s\" (VariableNode \"$%1$s\"))" +
                    ")", object, firstPredicate, secondPredicate);
        }
        return "";   
    }
    
    @Override
    public String toString() {
        return name + "(" + arguments.stream().map(fn -> fn.toString()).collect(Collectors.joining(", ")) + ")";
    }
}