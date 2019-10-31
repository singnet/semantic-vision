package atomspace;

import atomspace.query.ASQueryEngine.ASQueryResult;
import atomspace.storage.ASAtom;
import org.junit.Assert;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;

public class ASTestUtils {

    public static <T> void assertIteratorEquals(Iterator<T> iter, T... elems) {

        Set<T> expected = new HashSet<>();
        expected.addAll(Arrays.asList(elems));

        Set<T> actual = new HashSet<>();

        while (iter.hasNext()) {
            actual.add(iter.next());
        }


        Assert.assertEquals(expected, actual);
    }

    public static <T> void assertIteratorEqualsSequentially(Iterator<T> iter, T... elems) {

        List<T> expected = Arrays.asList(elems);
        List<T> actual = new ArrayList<>();

        while (iter.hasNext()) {
            actual.add(iter.next());
        }


        Assert.assertEquals(expected, actual);
    }

    public static void assertQueryResultsEqual(Iterator<ASQueryResult> actual, ASQueryResult... expect) {

        Set<ASQueryResult> actualSet = new HashSet<>();
        while (actual.hasNext()) {
            actualSet.add(actual.next());
        }

        Set<ASQueryResult> expectedSet = new HashSet<>();
        expectedSet.addAll(Arrays.asList(expect));

        Assert.assertEquals(expectedSet, actualSet);
    }

    public static <T> int count(Iterator<T> iter) {

        int count = 0;
        while (iter.hasNext()) {
            iter.next();
            count++;
        }

        return count;
    }

    public static <T, R> Iterator<R> map(Iterator<T> iter, Function<T, R> mapper) {
        return new Iterator<R>() {
            @Override
            public boolean hasNext() {
                return iter.hasNext();
            }

            @Override
            public R next() {
                return mapper.apply(iter.next());
            }
        };
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

    public static class KeyWithValue<K, V> {

        public final K key;
        public final V value;

        public KeyWithValue(K key, V value) {
            this.key = key;
            this.value = value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o instanceof KeyWithValue) {
                KeyWithValue<K, V> that = (KeyWithValue<K, V>) o;
                return Objects.equals(key, that.key) &&
                        Objects.equals(value, that.value);
            }
            return false;
        }

        @Override
        public int hashCode() {
            return Objects.hash(key, value);
        }
    }

    public static class TestQueryResult implements ASQueryResult {
        private final ASAtom atom;
        private final Map<String, ASAtom> variables = new HashMap<>();

        public TestQueryResult(ASAtom atoml, KeyWithValue<String, ASAtom>... variables) {
            this.atom = atoml;
            for (KeyWithValue<String, ASAtom> variable : variables) {
                this.variables.put(variable.key, variable.value);
            }
        }

        @Override
        public ASAtom getAtom() {
            return atom;
        }

        @Override
        public Map<String, ASAtom> getVariables() {
            return variables;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj instanceof ASQueryResult) {
                ASQueryResult that = (ASQueryResult) obj;
                return Objects.equals(this.getAtom(), that.getAtom()) &&
                        Objects.equals(this.getVariables(), that.getVariables());

            }
            return false;
        }

        @Override
        public int hashCode() {
            return Objects.hash(atom, variables);
        }

        @Override
        public String toString() {
            return String.format("atom: %s, variables: %s", atom, variables);
        }
    }

    public static String getTempDir(String prefix) throws IOException {
        return Files.createTempDirectory(prefix).toAbsolutePath().toString();
    }

    public static String getCleanNormalizedTempDir(String prefix) throws IOException {
        String path = getTempDir(prefix);
        removeDirectory(path);
        return path.replace('\\', '/');
    }
}
