# AtomspaceRunner

AtomspaceRunner is written in Kotlin and provides DSL style to work with Atomspace.

For example facts like 'Alice likes apple', 'Alice dislikes pears', and query 'What does Alice like?'
can be written as:
```kotlin
    val runner = SampleAtomspaceRunner(AtomspaceMemoryStorage())

    init(runner) {

        LikesLink(
                PersonNode("Alice"),
                ItemNode("apple")
        )

        DislikesLink(
                PersonNode("Alice"),
                ItemNode("pear")
        )
    }

    val results = query(runner) {
        LikesLink(
                PersonNode("Alice"),
                VariableNode("WHAT")
        )
    }

    results.variables("WHAT").nodes().forEach {
        println("Alice likes ${it.value}")
    }
```

Output:
```text
Alice likes apple
```

To make this example work it is necessary to extend  `AtomspaceRunner` in the following way:
```kotlin
class SampleAtomspaceRunner(override val atomspace: AtomspaceStorage) : AtomspaceRunner(atomspace) {

    fun PersonNode(value: String): ASAtom =
            tx.get("PersonNode", value)

    fun ItemNode(value: String): ASAtom =
            tx.get("ItemNode", value)

    fun LikesLink(person: ASAtom, item: ASAtom): ASAtom =
            tx.get("LikesLink", person, item)

    fun DislikesLink(person: ASAtom, item: ASAtom): ASAtom =
            tx.get("DislikesLink", person, item)
}
```