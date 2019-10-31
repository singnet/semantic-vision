# Atomspace Storage

Atomspace storage contains implementation for atoms which can be stored and queried by backing storage.
The main idea is that atoms stored in backing storage must be unique.

Each atom can be a node which has type and value or link which has type and list of outgoing atoms.

There are two main types of atoms: `RawAtom` and `ASAtom`.

`RawAtom` is an atom in memory which need not to be unique. It represents a user data which should be
stored in backing storage.

For example:
```java
        RawLink rawLink = new RawLink("Link1",
                new RawLink("Link2",
                        new RawNode("Node1", "value1"),
                        new RawNode("Node2", "value2")),
                new RawLink("Link3",
                        new RawNode("Node1", "value1")));
```
Note that both `Link2` and `Link3` contain the same `Node1` node with `value1` value which are represented
as two different nodes in memory.

`ASAtom` represents the atom in backing storage and the same atoms are unique in the backing storage.

For example:
```java
            RawLink rawLink = new RawLink("Link1",
                    new RawLink("Link2",
                            new RawNode("Node1", "value1"),
                            new RawNode("Node2", "value2")),
                    new RawLink("Link3",
                            new RawNode("Node1", "value1")));

            ASLink link = tx.get(rawLink);
```

The `ASLink` can be represented in storage like : `Link[2304]: Link1([3072, 4096])`
where the link type is `Link1`, its id in the backing storage is `2304` and its children ids are `[3072, 4096]`.

The dump of a backing storage can look like:
```text
Node[3328]: Node1(value1)
Node[3584]: Node2(value2)
Link[3072]: Link2([3328, 3584])
Link[4096]: Link3([3328])
Link[2304]: Link1([3072, 4096])
```

And indeed there is only one `Node1` with `value1` with id `3328` and both `Link2` and `Link3` point to the same
node with id `3328`.

# Atoms

## RawAtom

Raw atoms represent user`s data in memory which need to be loaded into the backing storage
and they are not unique.

Hierarchy of raw atoms:

RawAtom
* `type`

RawNode <- RawAtom
* `value`

RawLink <- RawAtom
* `atoms` : RawAtom[]

## ASAtom

AS atoms represent atoms in backing storage and they must be unique.

Each ASAtom has an unique id that allows to efficiently retrieve the atom from the backing storage.

AS atoms have incomingSet which consists of links that points to the given atom.
Incoming set is used by a query engine to retrieve atoms from the backing storage using atom patterns.

Hierarchy of AS atoms:

ASAtom
* `id`
* `type`
* `incomingSet`: ASIncomingSet

ASNode <- ASAtom
* `value`

ASLink <- ASAtom
* `outgoingList`: ASOutgoingList

* ASIncomingSet
  * `getIncomingSetSize(ASTransaction tx, String type, int arity, int position)`: `int`
  * `getIncomingSet(ASTransaction tx, String type, int arity, int position)`: `[id]`

* ASOutgoingList
  * `getArity(ASTransaction tx)`: `int`
  * `getAtom(ASTransaction tx, int index)` : ASAtom

# Backing Storage

## AtomspaceStorage

The AtomspaceStorage interface represents the backing storage where atoms are stored.
It contains the only method that returns the storage transaction. It is the transaction
that allows to store into and get atoms from the backing storage.

AtomspaceStorage
* `getTx()`: ASTransaction

## ASTransaction

ASTransaction allows to store and get atoms from the backing storage and has `commit` and `rallback` methods
which must be supported or emulated by the underlying backing storage.

ASTransaction has two types of methods: methods which work with `RawAtom`s and methods which work with `ASAtom`s.

Method that take `RawAtom` as argument need to traverse the whole atom tree and are used to initial data
loading into the backing storage.

Method that take `ASAtom` as argument can use the atom id and directly retrieve the atom and links children
from the backing storage.

ASTransaction
* RawAtom
    * `get(RawNode node)`: ASNode
    * `get(RawLink link)`: ASLink
    * `get(RawAtom atom)`: ASAtom
* ASAtom
    * `get(String type, String value)`: ASNode
    * `get(String type, ASAtom... atoms)`: ASLink
    * `get(long id)`: ASAtom
    * `getOutgoingListIds(long id)`: long[]
    * `getIncomingSetSize(long id, String type, int arity, int position)`: int
    * `getIncomingSet(long id, String type, int arity, int position)`: `Iterator<ASLink>`
* Get all atoms
    * `getAtoms()`: `Iterator<ASAtom> `
* Transaction
    * `commit()`
    * `rallback()`

Note that some methods which work with AS atoms are lazy so they do not load the whole link.
For example the method `get(long id)` loads only type and list of children ids for the link
and the real children are loaded lazily by demand.
