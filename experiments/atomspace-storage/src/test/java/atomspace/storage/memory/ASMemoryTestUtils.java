package atomspace.storage.memory;

public class ASMemoryTestUtils {

    public static AtomspaceMemoryStorage getTestStorage() {
        return new AtomspaceMemoryStorage();
    }

    public static AtomspaceMemoryStorageHelper getStorageHelper(AtomspaceMemoryStorage storage) {
        return new AtomspaceMemoryStorageHelper(storage);
    }
}