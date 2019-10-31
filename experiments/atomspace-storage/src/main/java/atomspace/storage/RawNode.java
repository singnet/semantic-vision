package atomspace.storage;

import java.util.Objects;

final public class RawNode extends RawAtom {

    final String value;

    public RawNode(String type, String value) {
        super(type);
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o instanceof RawNode) {
            RawNode that = (RawNode) o;
            return Objects.equals(this.type, that.type)
                    && Objects.equals(this.value, that.value);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, value);
    }

    @Override
    public String toString() {
        return String.format("%s('%s')", type, value);
    }
}
