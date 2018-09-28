package org.opencog.vqa.relex;

public interface ToQueryConverter {
    boolean isApplicable(RelexFormula formula);
    String getSchemeQuery(RelexFormula relexFormula);
    String getSchemeQueryURE(RelexFormula relexFormula);
    String getSchemeQueryPM(RelexFormula relexFormula);
    String getFullFormula();
    String getQuestionType();
}
