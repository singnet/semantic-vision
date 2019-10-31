package atomspace.storage;

public interface ASOutgoingList {

    int getArity(ASTransaction tx);

    ASAtom getAtom(ASTransaction tx, int index);

    default String toString(ASTransaction tx) {
        return toString();
    }

    default String toString(ASAtom... atoms) {

        StringBuilder builder = new StringBuilder();

        for (int i = 0; i < atoms.length; i++) {
            if (i != 0) {
                builder.append(',');
            }
            builder.append(atoms[i]);
        }

        return builder.toString();
    }
}
