package atomspace.storage;

public interface ASNode extends ASAtom {

    String getValue();

    default String toString(ASNode node) {
        return String.format("%s('%s')", node.getType(), node.getValue());
    }
}
