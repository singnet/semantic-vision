package atomspace.storage.memory;

import atomspace.storage.ASNode;
import atomspace.storage.ASTransaction;

public class ASMemoryNode extends ASMemoryAtom implements ASNode {

    private final String value;

    public ASMemoryNode(long id, String type, String value) {
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
