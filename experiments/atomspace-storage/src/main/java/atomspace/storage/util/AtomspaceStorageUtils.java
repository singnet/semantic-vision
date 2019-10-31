package atomspace.storage.util;

import atomspace.storage.ASAtom;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.Iterator;

public class AtomspaceStorageUtils {

    public static String getKey(String type, int arity, int position) {
        return String.format("%s_%d_%d", type, arity, position);
    }

    public static long[] toIds(ASAtom... atoms) {
        long[] ids = new long[atoms.length];

        for (int i = 0; i < atoms.length; i++) {
            ids[i] = atoms[i].getId();
        }
        return ids;
    }

    public static long[] toIds(String str) {
        String[] split = str.split(":");

        long[] ids = new long[split.length];

        for (int i = 0; i < split.length; i++) {
            ids[i] = Long.parseLong(split[i]);
        }
        return ids;
    }

    public static String idsToString(long... ids) {

        switch (ids.length) {
            case 0:
                return "";
            case 1:
                return Long.toString(ids[0]);
            default: {

                StringBuilder builder = new StringBuilder();
                for (long id : ids) {
                    builder.append(id).append(':');
                }
                return builder.toString();
            }
        }
    }

    public static <T> int count(Iterator<T> iter) {
        int s = 0;
        for (; iter.hasNext(); iter.next()) {
            s++;
        }
        return s;
    }

    public static void removeDirectory(String directory) {
        Path pathToBeDeleted = Paths.get(directory);

        if (!Files.exists(pathToBeDeleted)) {
            return;
        }

        try {
            Files.walk(pathToBeDeleted)
                    .sorted(Comparator.reverseOrder())
                    .map(Path::toFile)
                    .forEach(File::delete);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
