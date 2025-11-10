import scala.io.StdIn.readLine

object Levenshtein {
  def compute(s1: String, s2: String): Int = {
    val m = s1.length
    val n = s2.length
    val dp = Array.ofDim[Int](m + 1, n + 1)

    for (i <- 0 to m) dp(i)(0) = i
    for (j <- 0 to n) dp(0)(j) = j

    for (i <- 1 to m) {
      for (j <- 1 to n) {
        val cost = if (s1(i - 1) == s2(j - 1)) 0 else 1
        dp(i)(j) = Seq(
          dp(i - 1)(j) + 1,       // deletion
          dp(i)(j - 1) + 1,       // insertion
          dp(i - 1)(j - 1) + cost // substitution
        ).min
      }
    }

    dp(m)(n)
  }

  def main(args: Array[String]): Unit = {
    val s1 = readLine()
    val s2 = readLine()
    println(compute(s1, s2))
  }
}

