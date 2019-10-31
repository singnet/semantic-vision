package atomspace.storage;

import java.io.Closeable;
import java.util.Iterator;

/**
 * ASTransaction contains main methods to create and retrieve atoms
 * from the underlined storage.
 * <p>
 * Note: depending on the underlined storage some methods may not be
 * supported and throw UnsupportedOperationException
 *
 * @see UnsupportedOperationException
 */
public interface ASTransaction extends Closeable {

    /**
     * Returns ASNode which represents the node in the backing storage
     * for the given raw node.
     * Storages that allow to check if the node exists or create it otherwise
     * may override this method to retrieve the node in one request
     * for more efficiency.
     *
     * @param node raw node
     * @return node in the backing storage
     */
    default ASNode get(RawNode node) {
        return get(node.getType(), node.getValue());
    }

    /**
     * Returns ASLink which represents the link in the backing storage
     * for the given raw link.
     * Storages that allow to check if the link exists or create it otherwise
     * in one query may override this method to retrieve the node in one request
     * for more efficiency.
     *
     * @param link raw link
     * @return link in the backing storage
     */
    default ASLink get(RawLink link) {
        int arity = link.getArity();
        ASAtom[] atoms = new ASAtom[arity];

        for (int i = 0; i < arity; i++) {
            atoms[i] = get(link.getAtom(i));
        }
        return get(link.getType(), atoms);
    }

    /**
     * Returns ASAtom which represents the atom in the backing storage
     * for the given raw link.
     *
     * @param atom raw atom
     * @return atom in the backing storage
     */
    default ASAtom get(RawAtom atom) {
        if (atom instanceof RawNode) {
            return get((RawNode) atom);
        } else if (atom instanceof RawLink) {
            return get((RawLink) atom);
        } else {
            String msg = String.format("Unknown RawAtom class: %s", atom.getClass());
            throw new RuntimeException(msg);
        }
    }

    /**
     * Gets or creates Node by given type and value.
     *
     * @param type  the type of the atom
     * @param value the value of the atom
     * @return node in the backing storage
     */
    ASNode get(String type, String value);

    /**
     * Gets or creates Link by given type and outgoing list.
     *
     * @param type  the type of the atom
     * @param atoms the outgoing list of the atom
     * @return link in the backing storage
     */
    ASLink get(String type, ASAtom... atoms);

    /**
     * Gets the atom by for the unique identifier.
     *
     * @param id the id of the atom
     * @return atom in the backing storage
     */
    ASAtom get(long id);

    /**
     * Gets ids of the atom outgoing list by the given atom id
     *
     * @param id the id of the atom
     * @return outgoing list ids
     */
    long[] getOutgoingListIds(long id);

    /**
     * Gets size of the atom incoming set by the given id
     *
     * @param id       the id of the atom
     * @param type     the type of the atom
     * @param arity    the arity of the atom
     * @param position the index of the atom in the parent Link
     * @return size of the incoming set
     */
    int getIncomingSetSize(long id, String type, int arity, int position);

    /**
     * Gets the incoming set of the atom by the given id
     *
     * @param id       the id of the atom
     * @param type     the type of the atom
     * @param arity    the arity of the atom
     * @param position the index of the atom in the parent Link
     * @return incoming set
     */
    Iterator<ASLink> getIncomingSet(long id, String type, int arity, int position);

    /**
     * Returns all atoms
     *
     * @return all atoms from the underlined storage
     */
    Iterator<ASAtom> getAtoms();

    /**
     * Commits the current transaction
     */
    void commit();
}
