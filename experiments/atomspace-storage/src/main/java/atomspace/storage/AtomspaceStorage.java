package atomspace.storage;

public interface AtomspaceStorage extends AutoCloseable {

    ASTransaction getTx();
}
