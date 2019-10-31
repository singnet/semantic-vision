The main of the project is investigation of a simple AtomSpace implementation on top of different (including relational
and graph) backing storages.

# Description

Main modules:
* [Atomspace Storage](src/main/java/atomspace/storage)
* [Query Engine](src/main/java/atomspace/query)
* [Performance](src/main/java/atomspace/performance)

# Current status

What is implemented:
* Atoms creation
* Atoms querying

What is not implemented:
* Atoms removing
* Atoms key-value properties
* Child atomspaces
* Querying typed atoms
* Querying atoms which contains only variable nodes

Supported backing storages:
* InMemory
  * [Memory](src/main/java/atomspace/storage/memory)
* Relational DB
  * [Derby](src/main/java/atomspace/storage/relationaldb)
* Property Graph
  * [Neo4j](src/main/java/atomspace/storage/neo4j)
  * [JanusGraph](src/main/java/atomspace/storage/janusgraph)
* Gremlin
  * [Gremlin layer](src/main/java/atomspace/storage/layer/gremlin)

# Example

An example which demonstrates simple facts creation:
* Alice likes apple
* Alice likes orange
* Alice dislikes pear
* Bob likes apple

and queries:
* What does Alice like?
* Who likes apple?

```kotlin
    val runner = SampleAtomspaceRunner(AtomspaceMemoryStorage())

    init(runner) {

        LikesLink(
                PersonNode("Alice"),
                ItemNode("apple")
        )

        LikesLink(
                PersonNode("Alice"),
                ItemNode("orange")
        )

        DislikesLink(
                PersonNode("Alice"),
                ItemNode("pear")
        )

        LikesLink(
                PersonNode("Bob"),
                ItemNode("apple")
        )
    }

    // What does Alice like?
    val aliceLikes = query(runner) {
        LikesLink(
                PersonNode("Alice"),
                VariableNode("WHAT")
        )
    }

    println("What does Alice like?")
    aliceLikes.variables("WHAT").nodes().forEach {
        println("Alice likes ${it.value}.")
    }

    // Who likes apple?
    val likesApple = query(runner) {
        LikesLink(
                VariableNode("WHO"),
                ItemNode("apple")
        )
    }

    println("Who likes apples?")
    likesApple.variables("WHO").nodes().forEach {
        println("${it.value} likes apple.")
    }
```
The output is:
```text
What does Alice like?
Alice likes apple.
Alice likes orange.

Who likes apples?
Alice likes apple.
Bob likes apple.
```

For more details see [AtomspaceRunner](src/main/kotlin/atomspace/runner)

# Building

Build project:
```bash
gradle build
```

Run tests:
```bash
gradle test
```