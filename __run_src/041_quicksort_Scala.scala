// QuicksortInPlace.scala
// In-place quicksort on an array in Scala.
//
// • Reads integers (one per line) from a file path passed as the first CLI arg,
//   or from STDIN if no file is given.
// • Skips blank lines and lines starting with '#'.
// • Sorts in place using an iterative quicksort with Hoare partition
//   and a median-of-three pivot to reduce worst-case behavior.
// • Prints the sorted numbers, one per line, to STDOUT.
//
// Build & run (Scala 2 or 3):
//   scalac QuicksortInPlace.scala && scala QuicksortInPlace input.txt
//   # or
//   printf "5\n3\n8\n1\n2\n" | scala QuicksortInPlace

import scala.io.Source
import scala.util.Using
import scala.collection.mutable
import java.io.InputStreamReader

object QuicksortInPlace {

  def main(args: Array[String]): Unit = {
    try {
      val nums: Vector[Long] =
        if (args.nonEmpty) readNumbersFromFile(args(0))
        else readNumbersFromStdIn()

      if (nums.length <= 1) {
        nums.foreach(println)
        return
      }

      val arr = nums.toArray
      quickSortInPlace(arr)
      arr.foreach(println)
    } catch {
      case e: Throwable =>
        Console.err.println(s"error: ${e.getMessage}")
        sys.exit(1)
    }
  }

  // ---------- IO ----------
  private def readNumbersFromFile(path: String): Vector[Long] =
    Using.resource(Source.fromFile(path)) { src => readNumbersCore(src) }

  private def readNumbersFromStdIn(): Vector[Long] =
    Using.resource(Source.fromInputStream(System.in)) { src => readNumbersCore(src) }

  private def readNumbersCore(src: Source): Vector[Long] = {
    val buf = Vector.newBuilder[Long]
    var lineNo = 0
    for (raw <- src.getLines()) {
      lineNo += 1
      val line = raw.trim
      if (line.nonEmpty && !line.startsWith("#")) {
        try {
          buf += line.toLong
        } catch {
          case _: NumberFormatException =>
            throw new IllegalArgumentException(s"line $lineNo: not an integer: $line")
        }
      }
    }
    buf.result()
  }

  // ---------- Quicksort (iterative, Hoare partition, median-of-three pivot) ----------
  def quickSortInPlace(a: Array[Long]): Unit = {
    if (a.length < 2) return

    val stack = new mutable.ArrayStack[(Int, Int)]
    stack.push((0, a.length - 1))

    while (stack.nonEmpty) {
      val (lo, hi) = stack.pop()
      if (lo >= hi) {
        // nothing
      } else {
        val p = hoarePartition(a, lo, hi)

        // Process smaller partition first to keep stack shallow
        val leftSize  = p - lo + 1        // [lo..p]
        val rightSize = hi - (p + 1) + 1  // [p+1..hi]
        if (leftSize < rightSize) {
          if (p + 1 < hi) stack.push((p + 1, hi))
          if (lo < p)     stack.push((lo, p))
        } else {
          if (lo < p)     stack.push((lo, p))
          if (p + 1 < hi) stack.push((p + 1, hi))
        }
      }
    }
  }

  // Hoare partition using median-of-three pivot selection.
  private def hoarePartition(a: Array[Long], lo: Int, hi: Int): Int = {
    val mid = lo + ((hi - lo) >>> 1)
    val pivot = medianOfThree(a(lo), a(mid), a(hi))

    var i = lo - 1
    var j = hi + 1
    while (true) {
      do { i += 1 } while (a(i) < pivot)
      do { j -= 1 } while (a(j) > pivot)
      if (i >= j) return j
      swap(a, i, j)
    }
    j // unreachable; Scala wants an expression
  }

  private def medianOfThree(x0: Long, y0: Long, z0: Long): Long = {
    var x = x0; var y = y0; var z = z0
    if (x > y) { val t = x; x = y; y = t }
    if (y > z) { val t = y; y = z; z = t }
    if (x > y) { val t = x; x = y; y = t }
    y // median
  }

  private def swap(a: Array[Long], i: Int, j: Int): Unit = {
    val t = a(i); a(i) = a(j); a(j) = t
  }
}

