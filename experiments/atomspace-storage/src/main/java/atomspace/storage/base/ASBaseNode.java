package atomspace.storage.base;

import atomspace.storage.ASNode;
import atomspace.storage.ASTransaction;

public class ASBaseNode extends ASBaseAtom implements ASNode {

    final String value;

    public ASBaseNode(long id, String type, String value) {
        super(id, type);
        this.value = value;
    }

    @Override
    public String getValue() {
        return value;
    }

    @Override
    public String toString() {
        return toString(this);
    }

    @Override
    public String toString(ASTransaction tx) {
        return toString(this);
    }
}
