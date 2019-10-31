package atomspace.storage.base;

import atomspace.storage.ASLink;
import atomspace.storage.ASIncomingSet;
import atomspace.storage.ASTransaction;

import java.util.Iterator;

public class ASBaseIncomingSet implements ASIncomingSet {

    final long atomId;

    public ASBaseIncomingSet(long atomId) {
        this.atomId = atomId;
    }

    @Override
    public int getIncomingSetSize(ASTransaction tx, String type, int arity, int position) {
        return tx.getIncomingSetSize(atomId, type, arity, position);
    }

    @Override
    public Iterator<ASLink> getIncomingSet(ASTransaction tx, String type, int arity, int position) {
        return tx.getIncomingSet(atomId, type, arity, position);
    }
}
