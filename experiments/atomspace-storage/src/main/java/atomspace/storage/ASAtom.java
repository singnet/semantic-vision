package atomspace.storage;

public interface ASAtom {

    long getId();

    String getType();

    ASIncomingSet getIncomingSet();

    String toString(ASTransaction tx);
}
