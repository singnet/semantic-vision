package atomspace.storage.base;

import atomspace.storage.ASAtom;
import atomspace.storage.ASLink;
import atomspace.storage.ASOutgoingList;
import atomspace.storage.ASTransaction;

/**
 * Basic ASLink implementation which has constructor
 * for eager and lazy outgoing list initialization.
 *
 * @see ASBaseOutgoingList
 */
public class ASBaseLink extends ASBaseAtom implements ASLink {

    final ASOutgoingList outgoingList;

    public ASBaseLink(long id, String type, int arity) {
        super(id, type);
        this.outgoingList = new ASBaseOutgoingList(id, arity);
    }

    public ASBaseLink(long id, String type, long... ids) {
        super(id, type);
        this.outgoingList = new ASBaseOutgoingList(id, ids);
    }

    public ASBaseLink(long id, String type, ASAtom... atoms) {
        super(id, type);
        this.outgoingList = new ASBaseOutgoingList(id, atoms);
    }

    @Override
    public ASOutgoingList getOutgoingList() {
        return outgoingList;
    }

    @Override
    public String toString() {
        return toString(this);
    }

    @Override
    public String toString(ASTransaction tx) {
        StringBuilder builder = new StringBuilder()
                .append(getType())
                .append("(");

        ASOutgoingList outgoingList = getOutgoingList();
        for (int i = 0; i < outgoingList.getArity(tx); i++) {
            if (i > 0) {
                builder.append(',');
            }
            ASAtom atom = outgoingList.getAtom(tx, i);
            builder.append(atom.toString(tx));
        }

        builder.append(")");

        return builder.toString();
    }
}
