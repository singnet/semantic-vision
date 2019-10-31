package atomspace.storage;

public abstract class ASAbstractTest {

    protected abstract AtomspaceStorage getStorage();

    protected void testAtomspaceStorage(AtomspaceStorageTest test) throws Exception {

        AtomspaceStorage storage = getStorage();
        test.run(storage);
    }

    protected void testAtomspaceTransaction(AtomspaceTransactionTest test) throws Exception {

        AtomspaceStorage storage = getStorage();
        try (ASTransaction tx = storage.getTx()) {

            test.run(tx);
            tx.commit();
        }
    }

    protected interface AtomspaceStorageTest {

        void run(AtomspaceStorage storage) throws Exception;
    }

    protected interface AtomspaceTransactionTest {

        void run(ASTransaction tx) throws Exception;
    }

}
