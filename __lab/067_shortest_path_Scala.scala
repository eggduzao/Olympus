// Main.scala
// Breadth-First Search (BFS) shortest path on an unweighted, *undirected* graph.
//
// INPUT FORMAT (tab- or space-separated; one record per line):
//   Edge lines : "U  V1 [V2 ...]"   => add undirected edges U—V1, U—V2, ...
//   Query lines: "#  SRC  DST"      => request shortest path from SRC to DST
//
// Example (matches your prompt):
//   A   B   F
//   B   A   C
//   C   B   D
//   D   C   E
//   E   D   F
//   F   A   E
//   #   A   E
//
// USAGE
//   scala Main.scala < graph.tsv
//   # or with scalac:
//   scalac Main.scala && scala Main < graph.tsv
//   # or pass a file path:
//   scala Main.scala path/to/graph.tsv

import scala.io.Source
import scala.collection.mutable

object Main {
  def main(args: Array[String]): Unit = {
    val src: Source =
      if (args.nonEmpty) Source.fromFile(args(0))
      else Source.fromInputStream(System.in)

    val (adj, queries) = parseInput(src.getLines())
    src.close()

    if (queries.isEmpty) {
      println("""No queries found (expect lines like: "# SRC DST").""")
      return
    }

    for ((s, t) <- queries) {
      if (s == t) {
        println(s)
      } else if (!adj.contains(s)) {
        println(s"No path found from $s to $t (source not in graph)")
      } else if (!adj.contains(t)) {
        println(s"No path found from $s to $t (destination not in graph)")
      } else {
        bfsPath(adj, s, t) match {
          case Some(path) => println(path.mkString(" -> "))
          case None       => println(s"No path found from $s to $t")
        }
      }
    }
  }

  // ------------------------------- Parsing -------------------------------

  private def parseInput(lines: Iterator[String])
    : (Map[String, Vector[String]], Vector[(String, String)]) = {

    val adj = mutable.HashMap.empty[String, mutable.LinkedHashSet[String]]
    val queries = mutable.ArrayBuffer.empty[(String, String)]

    lines
      .map(_.trim)
      .filter(_.nonEmpty)
      .foreach { line =>
        val tok = splitWs(line)
        if (tok.isEmpty) ()
        else if (tok.head.startsWith("#")) {
          if (tok.length >= 3) queries += ((tok(1), tok(2)))
        } else {
          val u = tok.head
          ensureKey(adj, u)
          tok.tail.foreach { v =>
            ensureKey(adj, v)
            // undirected, deduped, stable order
            adj(u) += v
            adj(v) += u
          }
        }
      }

    // freeze into immutable with stable neighbor order
    val immAdj = adj.view.mapValues(set => set.toVector).toMap
    (immAdj, queries.toVector)
  }

  private def splitWs(s: String): Vector[String] = {
    // Split on any whitespace (spaces or tabs), compressing runs.
    val b = Vector.newBuilder[String]
    var i = 0
    while (i < s.length) {
      while (i < s.length && s.charAt(i).isWhitespace) i += 1
      if (i >= s.length) { /* done */ }
      else {
        val start = i
        i += 1
        while (i < s.length && !s.charAt(i).isWhitespace) i += 1
        b += s.substring(start, i)
      }
    }
    b.result()
  }

  private def ensureKey(
      adj: mutable.HashMap[String, mutable.LinkedHashSet[String]],
      u: String): Unit = {
    if (!adj.contains(u)) adj(u) = mutable.LinkedHashSet.empty[String]
  }

  // ------------------------------- BFS -----------------------------------

  // Returns shortest path [src, ..., dst] or None if unreachable.
  private def bfsPath(
      adj: Map[String, Vector[String]],
      src: String,
      dst: String): Option[List[String]] = {

    val q = mutable.Queue[String](src)
    val visited = mutable.HashSet[String](src)
    val parent = mutable.HashMap.empty[String, String]

    while (q.nonEmpty) {
      val u = q.dequeue()
      val nbrs = adj.getOrElse(u, Vector.empty)
      nbrs.foreach { v =>
        if (!visited(v)) {
          visited += v
          parent(v) = u
          if (v == dst) return Some(reconstruct(parent.toMap, src, dst))
          q.enqueue(v)
        }
      }
    }
    None
  }

  private def reconstruct(
      parent: Map[String, String],
      src: String,
      dst: String): List[String] = {
    val path = mutable.ListBuffer[String](dst)
    var cur = dst
    while (cur != src) {
      cur = parent(cur)
      path.prepend(cur)
    }
    path.toList
  }
}

