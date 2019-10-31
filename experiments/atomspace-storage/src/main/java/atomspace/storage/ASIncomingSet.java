package atomspace.storage;

import java.util.Iterator;

public interface ASIncomingSet {

    int getIncomingSetSize(ASTransaction tx, String type, int arity, int position);

    Iterator<ASLink> getIncomingSet(ASTransaction tx, String type, int arity, int position);
}
