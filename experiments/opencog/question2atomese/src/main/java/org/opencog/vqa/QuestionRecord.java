package org.opencog.vqa;

import lombok.Builder;
import lombok.ToString;

/**
 * Record class keeps information about one question and able to serialize/deserialize it as 
 * delimited list of fields. It is as simple as POJO with builder and serialization logic.
 */
@ToString
@Builder(toBuilder = true)
class QuestionRecord {

    private static final String FIELD_DELIMITER = "::";

    private final String questionId;
    private final String questionType;
    private final String question;
    private final String imageId;
    private final String answer;
    private final String shortFormula;
    private final String fullFormula;

    public String getQuestionType() {
        return questionType;
    }

    public String getQuestion() {
        return question;
    }
    
    public String getAnswer() {
        return answer;
    }

    public static QuestionRecord load(String string) {
        String[] fields = string.split(FIELD_DELIMITER);

        QuestionRecordBuilder builder = QuestionRecord.builder();
        
        builder
            .questionId(fields[0])
            .questionType(fields[1])
            .question(fields[2])
            .imageId(fields[3])
            .answer(fields[4])
            .shortFormula(fields.length > 5 ? fields[5] : "")
            .fullFormula(fields.length > 6 ? fields[6] : "");
            
        return builder.build();
    }

    public String save() {
        return questionId 
                + FIELD_DELIMITER + questionType 
                + FIELD_DELIMITER + question 
                + FIELD_DELIMITER + imageId
                + FIELD_DELIMITER + answer
                + FIELD_DELIMITER + shortFormula
                + FIELD_DELIMITER + fullFormula;
    }

}
