package atomspace.storage.memory;

import atomspace.storage.ASAtom;
import atomspace.storage.ASIncomingSet;
import atomspace.storage.ASLink;
import atomspace.storage.ASTransaction;

import java.util.*;

public abstract class ASMemoryAtom implements ASAtom {

    public final long id;
    public final String type;

    final ASIncomingSet incomingSet;

    public ASMemoryAtom(long id, String type) {
        this.id = id;
        this.type = type;
        this.incomingSet = new ASMemoryIncomingSet();
    }

    @Override
    public long getId() {
        return id;
    }

    @Override
    public String getType() {
        return type;
    }

    @Override
    public ASIncomingSet getIncomingSet() {
        return incomingSet;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }

        if (obj instanceof ASMemoryAtom) {
            ASMemoryAtom that = (ASMemoryAtom) obj;
            return this.id == that.id;
        }

        return false;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(id);
    }

    static class ASMemoryIncomingSet implements ASIncomingSet {

        private final Map<String, Set<ASLink>> map = new HashMap<>();

        public void add(ASLink link, int arity, int position) {
            getOrCreateSet(link.getType(), arity, position).add(link);
        }

        @Override
        public int getIncomingSetSize(ASTransaction tx, String type, int arity, int position) {
            Set<ASLink> set = getSet(type, arity, position);
            return (set == null) ? 0 : set.size();
        }

        @Override
        public Iterator<ASLink> getIncomingSet(ASTransaction tx, String type, int arity, int position) {
            Set<ASLink> set = getSet(type, arity, position);
            return (set == null) ? Collections.emptyIterator() : set.iterator();
        }

        private static String getKey(String type, int arity, int position) {
            return String.format("%s_%d_%d", type, arity, position);
        }

        private Set<ASLink> getSet(String type, int arity, int position) {
            return map.get(getKey(type, arity, position));
        }

        private Set<ASLink> getOrCreateSet(String type, int arity, int position) {
            return map.computeIfAbsent(getKey(type, arity, position), key -> new HashSet<>());
        }
    }
}
