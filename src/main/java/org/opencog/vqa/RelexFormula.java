package org.opencog.vqa;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import lombok.Builder;
import lombok.Singular;

@Builder
public class RelexFormula {
    
    @Singular private final List<RelexPredicate> predicates;

    public String getFullFormula() {
        predicates.sort(Comparator.naturalOrder());
        return predicates.stream().map(fn -> fn.toFormula()).collect(Collectors.joining(";"));
    }

    public String getShortFormula() {
        predicates.sort(Comparator.naturalOrder());
        return predicates.stream().map(fn -> fn.toShortFormula()).collect(Collectors.joining(";"));
    }

}
