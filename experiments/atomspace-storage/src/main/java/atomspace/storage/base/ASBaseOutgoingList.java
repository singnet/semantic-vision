package atomspace.storage.base;

import atomspace.storage.ASAtom;
import atomspace.storage.ASOutgoingList;
import atomspace.storage.ASTransaction;

/**
 * Default implementation of lazy ASOutgoingList.
 * <p>
 * The outgoing list is initialized eagerly with provided atoms during link creation.
 * <p>
 * Lazy outgoing list initialization prevents full link construction
 * and is used when the link is retrieved from an atom outgoing set.
 * <p>
 * Lazy initialization is useful for atoms querying when some link outgoing
 * atoms are not matched with the query.
 */
public class ASBaseOutgoingList implements ASOutgoingList {

    private final long id;
    private final int arity;
    private long[] ids;
    private ASAtom[] atoms;
    private boolean isInitialized = false;

    public ASBaseOutgoingList(long id, int arity) {
        this.id = id;
        this.arity = arity;
    }

    public ASBaseOutgoingList(long id, long... ids) {
        this(id, ids.length);
        this.ids = ids;
        this.atoms = new ASAtom[ids.length];
        this.isInitialized = true;
    }

    public ASBaseOutgoingList(long id, ASAtom... atoms) {
        this(id, atoms.length);
        this.atoms = atoms;
        this.isInitialized = true;
    }

    @Override
    public int getArity(ASTransaction tx) {
        return arity;
    }

    @Override
    public ASAtom getAtom(ASTransaction tx, int index) {
        if (!isInitialized) {
            ids = tx.getOutgoingListIds(id);
            atoms = new ASAtom[arity];
            isInitialized = true;
        }

        ASAtom atom = atoms[index];

        if (atom == null) {
            long childId = ids[index];
            atoms[index] = atom = tx.get(childId);
        }

        return atom;
    }

    @Override
    public String toString() {
        return toString(atoms);
    }
}
