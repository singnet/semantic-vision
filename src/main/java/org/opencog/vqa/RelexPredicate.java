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

    @Override
    public String toString() {
        return name + "(" + arguments.stream().map(fn -> fn.toString()).collect(Collectors.joining(", ")) + ")";
    }
}