// BinarySearch.scala
// Scala CLI: Binary search (first occurrence) on a sorted ascending array of integers.
//
// Build/Run (Scala 2.13 or 3.x):
//   scalac BinarySearch.scala && scala BinarySearch <target> [path/to/file]
//   # or via STDIN (one number per line):
//   printf "1\n3\n4\n7\n9\n11\n15\n" | scala BinarySearch 11
//
// Behavior
//   • O(log N) search; returns the FIRST index if duplicates exist (stable-left).
//   • On hit : prints  FOUND <target> at index <i> (1-based)
//   • On miss: prints  NOT FOUND <target>. Insertion index <i> (1-based), between <left> and <right>
//   • Validates non-decreasing order; exits with error if violated.

import scala.io.Source
import scala.util.Try

object BinarySearch {
  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      Console.err.println("ERROR: missing <target>")
      usage()
      sys.exit(2)
    }

    val target: Long = Try(args(0).toLong).getOrElse {
      Console.err.println(s"ERROR: target must be integer, got '${args(0)}'")
      usage(); sys.exit(2); 0L
    }

    val numbers: Vector[Long] =
      try {
        if (args.length >= 2) readNumbers(Source.fromFile(args(1)))
        else readNumbers(Source.stdin)
      } catch {
        case e: Exception =>
          Console.err.println(s"ERROR: failed to read input: ${e.getMessage}")
          sys.exit(1); Vector.empty
      }

    if (numbers.isEmpty) {
      Console.err.println("ERROR: no numeric input provided.")
      sys.exit(1)
    }

    ensureAscending(numbers) match {
      case Some(err) =>
        Console.err.println(err)
        sys.exit(1)
      case None => // ok
    }

    val (idx1, found, ins1) = binarySearchFirst(numbers, target)

    if (found) {
      println(s"FOUND $target at index $idx1 (1-based)")
    } else {
      val left  = if (ins1 >= 2) numbers(ins1 - 2).toString else "-inf"
      val right = if ((ins1 - 1) < numbers.length) numbers(ins1 - 1).toString else "+inf"
      println(s"NOT FOUND $target. Insertion index $ins1 (1-based), between $left and $right")
    }
  }

  /** Read integers (one per line). Non-numeric lines are skipped with a warning. */
  def readNumbers(src: Source): Vector[Long] = {
    try {
      val buf = Vector.newBuilder[Long]
      var lineNo = 0
      for (line <- src.getLines()) {
        lineNo += 1
        val s = line.trim
        if (s.nonEmpty) {
          Try(s.toLong).toOption match {
            case Some(v) => buf += v
            case None    => Console.err.println(s"WARN: skipping non-integer line $lineNo: $s")
          }
        }
      }
      buf.result()
    } finally src.close()
  }

  /** Ensure non-decreasing order; return Some(errorMsg) if violated. */
  def ensureAscending(a: Vector[Long]): Option[String] = {
    var i = 1
    while (i < a.length) {
      if (a(i) < a(i - 1))
        return Some(s"ERROR: input not in ascending order at position ${i + 1}: ${a(i)} < ${a(i - 1)}")
      i += 1
    }
    None
  }

  /** Binary search for first occurrence (stable-left).
    * Returns (index_1based, found, insertion_index_1based).
    */
  def binarySearchFirst(a: Vector[Long], target: Long): (Int, Boolean, Int) = {
    var lo = 0
    var hi = a.length - 1
    var found = -1

    while (lo <= hi) {
      val mid = lo + (hi - lo) / 2
      val v = a(mid)
      if (v == target) {
        found = mid
        hi = mid - 1 // keep searching left
      } else if (v < target) {
        lo = mid + 1
      } else {
        hi = mid - 1
      }
    }

    if (found >= 0) (found + 1, true, lo + 1)
    else (0, false, lo + 1)
  }

  def usage(): Unit = {
    Console.err.println("Usage:")
    Console.err.println("  scala BinarySearch <target> [path/to/file]")
    Console.err.println("""  printf "1\n3\n4\n7\n9\n11\n15\n" | scala BinarySearch 11""")
  }
}
