package org.opencog.vqa.relex;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import lombok.Builder;
import lombok.Singular;
import relex.ParsedSentence;

public class RelexFormula {

    private final List<RelexPredicate> predicates;
    private final ParsedSentence relexSentence;

    @Builder
    public RelexFormula(@Singular List<RelexPredicate> predicates, ParsedSentence relexSentence) {
        this.predicates = sortRelexPredicates(predicates);
        this.relexSentence = relexSentence;
    }

    public List<RelexPredicate> getPredicates() {
        return Collections.unmodifiableList(predicates);
    }
    
    public ParsedSentence getRelexSentence() {
        return relexSentence;
    }

    public String getFullFormula() {
        return predicates.stream().map(fn -> fn.toFormula()).collect(Collectors.joining(";"));
    }

    public String getShortFormula() {
        return predicates.stream().map(fn -> fn.toShortFormula()).collect(Collectors.joining(";"));
    }

    public String getGroundedFormula() {
        return predicates.stream().map(fn -> fn.toGroundedFormula()).collect(Collectors.joining(";"));
    }
    
    public String getAtomeseFormula() {
        return "(SatisfactionLink (TypedVariableLink (VariableNode \"$A\") (TypeNode \"ConceptNode\")) (AndLink " + predicates.stream().map(fn -> fn.toAtomeseFormula()).collect(Collectors.joining(" "))
                + "))";
    }
    
    @Override
    public String toString() {
        return getFullFormula();
    }

    private List<RelexPredicate> sortRelexPredicates(List<RelexPredicate> predicates) {
        List<RelexPredicate> sorted = new ArrayList<>();
        sorted.addAll(predicates);
        sorted.sort(Comparator.naturalOrder());
        return Collections.unmodifiableList(sorted);
    }

}
