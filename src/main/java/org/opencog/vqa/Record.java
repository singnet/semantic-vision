package org.opencog.vqa;

class Record {
	
	private static final String FIELD_DELIMITER = ":";
	
	private final String questionId;
	private final String questionType;
	private final String question;
	private final String imageId;
	
	private Record(String questionId, String questionType, String question, String imageId) {
		this.questionId = questionId;
		this.questionType = questionType;
		this.question = question;
		this.imageId = imageId;
	}
	
	public String getQuestionType() {
		return questionType;
	}
	
	public String getQuestion() {
		return question;
	}

	public static Record parseRecord(String string) {
		String[] fields = string.split(FIELD_DELIMITER);

		String questionId = fields[0];
		String questionType = fields[1];
		String question = fields[2];
		String imageId = fields[3];
		
		return new Record(questionId, questionType, question, imageId);
	}
	
	public String printRecord() {
		return questionId + FIELD_DELIMITER + questionType + FIELD_DELIMITER + question + FIELD_DELIMITER + imageId;
	}

	@Override
	public String toString() {
		return "Record [questionId=" + questionId + ", questionType=" + questionType + ", question=" + question + ", imageId=" + imageId + "]";
	}
	
}
