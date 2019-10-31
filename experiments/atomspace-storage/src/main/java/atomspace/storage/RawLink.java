package atomspace.storage;

import java.util.Arrays;
import java.util.Objects;

final public class RawLink extends RawAtom {

    final RawAtom[] atoms;

    public RawLink(String type, RawAtom... atoms) {
        super(type);
        this.atoms = atoms;
    }

    public int getArity() {
        return atoms.length;
    }

    public RawAtom getAtom(int index) {
        return atoms[index];
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o instanceof RawLink) {
            RawLink that = (RawLink) o;
            return Objects.equals(this.type, that.type)
                    && Arrays.equals(this.atoms, that.atoms);
        }
        return false;
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(type);
        result = 31 * result + Arrays.hashCode(atoms);
        return result;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder()
                .append(type)
                .append("(");

        for (int i = 0; i < atoms.length; i++) {
            if (i != 0) {
                builder.append(',');
            }
            builder.append(atoms[i]);
        }

        builder.append(")");
        return builder.toString();
    }
}
