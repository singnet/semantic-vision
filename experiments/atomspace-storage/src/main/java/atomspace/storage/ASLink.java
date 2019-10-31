package atomspace.storage;

public interface ASLink extends ASAtom {

    ASOutgoingList getOutgoingList();

    default String toString(ASLink link) {

        return new StringBuilder()
                .append(link.getType())
                .append("(")
                .append(getOutgoingList())
                .append(")")
                .toString();
    }
}
