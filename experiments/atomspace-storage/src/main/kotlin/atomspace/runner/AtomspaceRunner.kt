package atomspace.runner

import atomspace.storage.*
import atomspace.query.ASQueryEngine.ASQueryResult
import atomspace.query.basic.ASBasicQueryEngine

open class AtomspaceRunner(open val atomspace: AtomspaceStorage) {

    val queryEngine = ASBasicQueryEngine()
    lateinit var tx: ASTransaction

    fun openTx(): ASTransaction {
        this.tx = atomspace.tx
        return tx
    }

    fun VariableNode(value: String): ASAtom =
            tx.get(ASBasicQueryEngine.TYPE_NODE_VARIABLE, value)

    fun dump() {
        for (atom in tx.atoms) {
            println(atom)
        }
    }
}

fun <ASRunner : AtomspaceRunner> init(runner: ASRunner, block: ASRunner.() -> Unit) {
    val tx = runner.openTx()
    runner.tx = tx
    runner.block()
    tx.commit()
    tx.close()
}

fun <ASRunner : AtomspaceRunner> query(runner: ASRunner, block: ASRunner.() -> ASAtom): Sequence<ASQueryResult> {
    val tx = runner.openTx()
    val query = runner.block()
    val results = runner.queryEngine.match(tx, query)
    tx.commit()
    tx.close()
    return results.asSequence()
}

fun Sequence<ASAtom>.nodes(): Sequence<ASNode> = this.filterIsInstance(ASNode::class.java)
fun Sequence<ASAtom>.links(): Sequence<ASLink> = this.filterIsInstance(ASLink::class.java)
fun Sequence<ASQueryResult>.variables(variable: String): Sequence<ASAtom> =
        this.map { it.variables[variable]!! }
