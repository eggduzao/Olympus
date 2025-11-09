levenshtein_distance <- function(s1, s2) {
  m <- nchar(s1)
  n <- nchar(s2)
  
  # Initialize DP matrix
  dp <- matrix(0, nrow = m + 1, ncol = n + 1)
  
  for (i in 0:m) {
    dp[i + 1, 1] <- i
  }
  for (j in 0:n) {
    dp[1, j + 1] <- j
  }
  
  # Fill DP
  for (i in 1:m) {
    for (j in 1:n) {
      cost <- ifelse(substr(s1, i, i) == substr(s2, j, j), 0, 1)
      dp[i + 1, j + 1] <- min(
        dp[i, j + 1] + 1,     # Deletion
        dp[i + 1, j] + 1,     # Insertion
        dp[i, j] + cost       # Substitution
      )
    }
  }
  
  return(dp[m + 1, n + 1])
}

# Read input
input <- readLines("stdin")
s1 <- trimws(input[1])
s2 <- trimws(input[2])

cat(levenshtein_distance(s1, s2), "\n")

