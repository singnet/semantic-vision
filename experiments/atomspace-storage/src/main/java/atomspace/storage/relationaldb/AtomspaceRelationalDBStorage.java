package atomspace.storage.relationaldb;

import atomspace.storage.AtomspaceStorage;

import java.sql.*;

public class AtomspaceRelationalDBStorage implements AtomspaceStorage {


    static final String TABLE_ATOMS = "ATOMS";
    static final String TABLE_INCOMING_SET = "INCOMING_SET";

    static final String CREATE_TABLE_ATOMS = String.format(
            "CREATE TABLE %s(" +
                    "id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY PRIMARY KEY," +
                    "type VARCHAR(255)," +
                    "value VARCHAR(255)," +
                    "arity INTEGER NOT NULL," +
                    "ids VARCHAR(1024)" + // Use LONG VARCHAR
                    ")",
            TABLE_ATOMS);

    static final String CREATE_TABLE_INCOMING_SET = String.format(
            "CREATE TABLE %s(" +
                    "id BIGINT," +
                    "type_arity_pos VARCHAR(255)," +
                    "parent_id BIGINT," +
                    "PRIMARY KEY (id, type_arity_pos, parent_id))",
            TABLE_INCOMING_SET);

    static final String CREATE_INDEX_ATOM = String.format(
            "CREATE UNIQUE INDEX atoms_index on %s(type, value, arity, ids)",
            TABLE_ATOMS);

    static final String CREATE_INDEX_INCOMING_SET = String.format(
            "CREATE INDEX incoming_sets_index on %s(id, type_arity_pos)",
            TABLE_INCOMING_SET);

    final Connection connection;

    public AtomspaceRelationalDBStorage(String dbURL) throws SQLException {
        this.connection = DriverManager.getConnection(dbURL);
        this.initDB();
    }

    @Override
    public ASRelationalDBTransaction getTx() {
        return new ASRelationalDBTransaction(connection);
    }

    @Override
    public void close() throws SQLException {
        connection.close();
    }

    void initDB() throws SQLException {

        initTable(TABLE_ATOMS, CREATE_TABLE_ATOMS);
        initTable(TABLE_INCOMING_SET, CREATE_TABLE_INCOMING_SET);

        initIndices();
    }

    private void initTable(String tableName, String sql) throws SQLException {
        DatabaseMetaData dbmd = connection.getMetaData();
        try (ResultSet rs = dbmd.getTables(null, null, tableName, null)) {
            if (!rs.next()) {
                try (Statement statement = connection.createStatement()) {
                    statement.execute(sql);
                }
            }
        }
    }

    private void initIndices() throws SQLException {
        try (Statement statement = connection.createStatement()) {
            statement.execute(CREATE_INDEX_ATOM);
            statement.execute(CREATE_INDEX_INCOMING_SET);
        }
    }
}
