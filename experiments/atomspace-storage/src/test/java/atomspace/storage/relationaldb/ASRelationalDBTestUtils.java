package atomspace.storage.relationaldb;

import atomspace.ASTestUtils;
import atomspace.storage.util.AtomspaceStorageHelper;

import java.io.IOException;
import java.sql.SQLException;

public class ASRelationalDBTestUtils {

    public static final String DB_URL_JUNIT = "jdbc:derby:%s;create=true";
    private static final AtomspaceRelationalDBStorage RELATIONALDB_STORAGE_STORAGE;

    static {
        try {
            String path = ASTestUtils.getCleanNormalizedTempDir("atomspace-storage-junit-relationaldb");
            String dbURL = String.format(DB_URL_JUNIT, path);
            RELATIONALDB_STORAGE_STORAGE = new AtomspaceRelationalDBStorage(dbURL);
        } catch (SQLException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static AtomspaceRelationalDBStorage getTestStorage() {
        resetStorage();
        return RELATIONALDB_STORAGE_STORAGE;
    }

    private static void resetStorage() {
        try (ASRelationalDBTransaction tx = RELATIONALDB_STORAGE_STORAGE.getTx()) {
            AtomspaceStorageHelper helper = new AtomspaceRelationalDBStorageHelper(RELATIONALDB_STORAGE_STORAGE);
            helper.reset(tx);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
