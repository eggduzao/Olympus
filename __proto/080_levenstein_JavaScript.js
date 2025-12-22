function levenshteinDistance(s1, s2) {
  const m = s1.length;
  const n = s2.length;

  // Initialize DP table
  const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));

  // Base cases
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  // Fill table
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,        // Deletion
        dp[i][j - 1] + 1,        // Insertion
        dp[i - 1][j - 1] + cost  // Substitution
      );
    }
  }

  return dp[m][n];
}

// Read input
const fs = require("fs");
const input = fs.readFileSync(0, "utf-8").trim().split("\n");
const s1 = input[0].trim();
const s2 = input[1].trim();

console.log(levenshteinDistance(s1, s2));

