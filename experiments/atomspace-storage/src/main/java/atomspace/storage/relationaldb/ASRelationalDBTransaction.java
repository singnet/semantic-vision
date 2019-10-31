package atomspace.storage.relationaldb;

import atomspace.storage.ASAtom;
import atomspace.storage.ASLink;
import atomspace.storage.ASNode;
import atomspace.storage.ASTransaction;
import atomspace.storage.base.ASBaseLink;
import atomspace.storage.base.ASBaseNode;
import atomspace.storage.util.AtomspaceStorageUtils;

import java.sql.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static atomspace.storage.relationaldb.AtomspaceRelationalDBStorage.TABLE_ATOMS;
import static atomspace.storage.relationaldb.AtomspaceRelationalDBStorage.TABLE_INCOMING_SET;
import static atomspace.storage.util.AtomspaceStorageUtils.*;

public class ASRelationalDBTransaction implements ASTransaction {

    static final String ID = "id";
    static final String TYPE = "type";
    static final String VALUE = "value";
    static final String ARITY = "arity";
    static final String IDS = "ids";

    static final String QUERY_ATOM_ID = String.format(
            "SELECT id from %s where type = ? and value = ? and arity = ? and ids = ?",
            TABLE_ATOMS);

    static final String QUERY_ATOM = String.format(
            "SELECT type, value, arity, ids from %s where id = ?",
            TABLE_ATOMS);

    static final String INSERT_ATOM = String.format(
            "INSERT INTO %s (type, value, arity, ids) values (?, ?, ?, ?)",
            TABLE_ATOMS);

    static final String QUERY_IDS = String.format(
            "SELECT ids from %s where id = ?",
            TABLE_ATOMS);


    static final String UPDATE_INCOMING_SET = String.format(
            "INSERT INTO %s (id, type_arity_pos, parent_id) values (?, ?, ?)",
            TABLE_INCOMING_SET);

    static final String QUERY_INCOMING_SET_SIZE = String.format(
            "SELECT count(*) as total from %s where id = ? and type_arity_pos = ?",
            TABLE_INCOMING_SET);

    static final String QUERY_INCOMING_SET = String.format(
            "SELECT parent_id from %s where id = ? and type_arity_pos = ?",
            TABLE_INCOMING_SET);

    static final String QUERY_ALL_ATOMS = String.format(
            "SELECT id, type, value, arity, ids from %s",
            TABLE_ATOMS);

    static final String COUNT_NODES = String.format(
            "SELECT count(*) as total from %s where arity = -1",
            TABLE_ATOMS);

    static final String COUNT_LINKS = String.format(
            "SELECT count(*) as total from %s where arity <> -1",
            TABLE_ATOMS);


    final Connection connection;

    public ASRelationalDBTransaction(Connection connection) {
        this.connection = connection;
    }

    @Override
    public ASNode get(String type, String value) {
        long id = get(type, value, -1);
        return new ASBaseNode(id, type, value);
    }

    @Override
    public ASLink get(String type, ASAtom... atoms) {
        long[] ids = AtomspaceStorageUtils.toIds(atoms);
        long id = get(type, "", atoms.length, ids);
        return new ASBaseLink(id, type, atoms);
    }

    private long get(String type, String value, int arity, long... ids) {

        try (PreparedStatement statement = connection.prepareStatement(QUERY_ATOM_ID)) {

            statement.setString(1, type);
            statement.setString(2, value);
            statement.setInt(3, arity);
            statement.setString(4, idsToString(ids));

            try (ResultSet resultSet = statement.executeQuery()) {

                if (resultSet.next()) {
                    long id = resultSet.getLong("id");
                    return id;
                }
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        try (PreparedStatement statement = connection
                .prepareStatement(INSERT_ATOM, Statement.RETURN_GENERATED_KEYS)) {

            statement.setString(1, type);
            statement.setString(2, value);
            statement.setInt(3, arity);
            statement.setString(4, idsToString(ids));

            statement.executeUpdate();

            try (ResultSet generatedKeys = statement.getGeneratedKeys()) {
                generatedKeys.next();
                long id = generatedKeys.getLong(1);

                if (isLink(arity)) {
                    updateIncomingSet(id, type, ids);
                }

                return id;
            }

        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private void updateIncomingSet(long parentId, String type, long... ids) {

        try (PreparedStatement statement = connection.prepareStatement(UPDATE_INCOMING_SET)) {

            int arity = ids.length;
            for (int i = 0; i < arity; i++) {

                long id = ids[i];
                String key = getKey(type, arity, i);

                statement.setLong(1, id);
                statement.setString(2, key);
                statement.setLong(3, parentId);

                statement.addBatch();
            }

            statement.executeBatch();

        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public ASAtom get(long id) {
        try (PreparedStatement statement = connection.prepareStatement(QUERY_ATOM)) {

            statement.setLong(1, id);
            try (ResultSet resultSet = statement.executeQuery()) {
                if (resultSet.next()) {

                    String type = resultSet.getString(1);
                    int arity = resultSet.getInt(3);

                    if (isNode(arity)) {
                        String value = resultSet.getString(VALUE);
                        return new ASBaseNode(id, type, value);
                    } else {
                        String childIds = resultSet.getString(IDS);
                        long[] ids = AtomspaceStorageUtils.toIds(childIds);
                        return new ASBaseLink(id, type, ids);
                    }
                }

                String msg = String.format("Atom with id %d was not found!", id);
                throw new RuntimeException(msg);
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public long[] getOutgoingListIds(long id) {

        try (PreparedStatement statement = connection.prepareStatement(QUERY_IDS)) {

            statement.setLong(1, id);
            try (ResultSet resultSet = statement.executeQuery()) {
                if (resultSet.next()) {

                    String childIds = resultSet.getString(IDS);
                    return AtomspaceStorageUtils.toIds(childIds);
                }

                String msg = String.format("Atom with id %d was not found!", id);
                throw new RuntimeException(msg);
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int getIncomingSetSize(long id, String type, int arity, int position) {
        try (PreparedStatement statement = connection.prepareStatement(QUERY_INCOMING_SET_SIZE)) {

            String key = getKey(type, arity, position);
            statement.setLong(1, id);
            statement.setString(2, key);
            try (ResultSet resultSet = statement.executeQuery()) {
                if (resultSet.next()) {
                    return resultSet.getInt(1);
                }
                return 0;
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Iterator<ASLink> getIncomingSet(long id, String type, int arity, int position) {
        List<ASLink> links = new ArrayList<>();
        try (PreparedStatement statement = connection.prepareStatement(QUERY_INCOMING_SET)) {

            String key = AtomspaceStorageUtils.getKey(type, arity, position);
            statement.setLong(1, id);
            statement.setString(2, key);
            try (ResultSet resultSet = statement.executeQuery()) {
                while (resultSet.next()) {
                    long parentId = resultSet.getLong(1);
                    links.add(new ASBaseLink(parentId, type, arity));
                }
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return links.iterator();
    }

    @Override
    public Iterator<ASAtom> getAtoms() {

        try (Statement statement = connection.createStatement()) {
            try (ResultSet resultSet = statement.executeQuery(QUERY_ALL_ATOMS)) {

                List<ASAtom> atoms = new ArrayList<>();

                if (resultSet.next()) {
                    long id = resultSet.getLong(ID);
                    String type = resultSet.getString(TYPE);
                    int arity = resultSet.getInt(ARITY);

                    if (isNode(arity)) {
                        String value = resultSet.getString(VALUE);
                        ASNode node = new ASBaseNode(id, type, value);
                        atoms.add(node);
                    } else {
                        String ids = resultSet.getString(IDS);
                        ASLink node = new ASBaseLink(id, type, toIds(ids));
                        atoms.add(node);
                    }
                }

                return atoms.iterator();
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private static boolean isNode(int arity) {
        return arity == -1;
    }

    private static boolean isLink(int arity) {
        return !isNode(arity);
    }

    @Override
    public void commit() {

    }

    @Override
    public void close() {
    }

    void reset() throws SQLException {
        resetTable(TABLE_ATOMS);
        resetTable(TABLE_INCOMING_SET);
    }

    void resetTable(String table) throws SQLException {

        String sql = String.format("TRUNCATE TABLE %s", table);

        DatabaseMetaData dbmd = connection.getMetaData();
        try (ResultSet rs = dbmd.getTables(null, null, table, null)) {
            if (rs.next()) {
                try (Statement statement = connection.createStatement()) {
                    statement.executeUpdate(sql);
                }
            }
        }
    }

    void printStatics(String msg) throws SQLException {
        int nodes = count(COUNT_NODES);
        long links = count(COUNT_LINKS);
        System.out.printf("%s nodes: %s, links: %s%n", msg, nodes, links);
    }

    public int count(String sql) throws SQLException {
        try (Statement statement = connection.createStatement()) {

            try (ResultSet resultSet = statement.executeQuery(sql)) {
                if (resultSet.next()) {
                    return resultSet.getInt(1);
                }
                return 0;
            }
        }
    }


    void dump(String msg, Object... args) {
        System.out.printf(msg, args);
        try {
            dump();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    void dump() throws SQLException {

        System.out.printf("--- Dump ---%n");
        dumpTable(TABLE_ATOMS);
        dumpTable(TABLE_INCOMING_SET);
        System.out.printf("--- ---- ---%n");
    }

    void dumpTable(String table) throws SQLException {

        String sql = String.format("select * from %s", table);

        System.out.printf("Table: %s%n", table);
        try (Statement st = connection.createStatement();
             ResultSet rs = st.executeQuery(sql)) {

            ResultSetMetaData rsmd = rs.getMetaData();
            int columnsNumber = rsmd.getColumnCount();

            while (rs.next()) {
                for (int i = 1; i <= columnsNumber; i++) {
                    String cell = rs.getObject(i).toString();
                    cell = cell.isEmpty() ? "''" : cell;
                    System.out.printf("%s ", cell);
                }
                System.out.println();
            }
        }
    }
}
