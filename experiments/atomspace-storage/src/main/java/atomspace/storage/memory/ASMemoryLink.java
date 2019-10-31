package atomspace.storage.memory;

import atomspace.storage.ASAtom;
import atomspace.storage.ASLink;
import atomspace.storage.ASOutgoingList;
import atomspace.storage.ASTransaction;

public class ASMemoryLink extends ASMemoryAtom implements ASLink {

    final ASOutgoingList outgoingList;

    public ASMemoryLink(long id, String type, ASAtom... atoms) {
        super(id, type);
        this.outgoingList = new ASMemoryOutgoingList(atoms);
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
        return toString(this);
    }

    static class ASMemoryOutgoingList implements ASOutgoingList {
        final ASAtom[] atoms;

        public ASMemoryOutgoingList(ASAtom[] atoms) {
            this.atoms = atoms;
        }

        @Override
        public int getArity(ASTransaction tx) {
            return atoms.length;
        }

        @Override
        public ASAtom getAtom(ASTransaction tx, int index) {
            return atoms[index];
        }

        @Override
        public String toString() {
            return toString(atoms);
        }
    }
}
