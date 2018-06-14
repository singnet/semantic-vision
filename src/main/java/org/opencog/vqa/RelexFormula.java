package org.opencog.vqa;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import lombok.Builder;
import lombok.Singular;


public class RelexFormula {
    
    private final List<RelexPredicate> predicates;

    @Builder
    public RelexFormula(@Singular List<RelexPredicate> predicates) {
        this.predicates = sortRelexPredicates(predicates);
    }

    public String getFullFormula() {
        return predicates.stream().map(fn -> fn.toFormula()).collect(Collectors.joining(";"));
    }

    public String getShortFormula() {
        return predicates.stream().map(fn -> fn.toShortFormula()).collect(Collectors.joining(";"));
    }

    private List<RelexPredicate> sortRelexPredicates(List<RelexPredicate> predicates) {
        List<RelexPredicate> sorted = new ArrayList<>();
        sorted.addAll(predicates);
        sorted.sort(Comparator.naturalOrder());
        return Collections.unmodifiableList(sorted);
    }

}
