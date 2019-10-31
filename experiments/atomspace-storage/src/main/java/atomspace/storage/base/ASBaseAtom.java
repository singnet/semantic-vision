package atomspace.storage.base;

import atomspace.storage.ASAtom;
import atomspace.storage.ASIncomingSet;

public abstract class ASBaseAtom implements ASAtom {

    final long id;
    final String type;
    final ASIncomingSet incomingSet;

    public ASBaseAtom(long id, String type) {
        this.id = id;
        this.type = type;
        this.incomingSet = new ASBaseIncomingSet(id);
    }

    @Override
    public long getId() {
        return id;
    }

    @Override
    public String getType() {
        return type;
    }

    @Override
    public ASIncomingSet getIncomingSet() {
        return incomingSet;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }

        if (obj instanceof ASAtom) {
            ASAtom that = (ASAtom) obj;
            return this.getId() == that.getId();
        }

        return false;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(getId());
    }
}
