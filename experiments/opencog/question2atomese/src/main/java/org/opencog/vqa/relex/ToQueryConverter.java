package org.opencog.vqa.relex;

public interface ToQueryConverter {
    
    boolean isApplicable(RelexFormula formula);
    String getSchemeQuery(RelexFormula relexFormula);
}
