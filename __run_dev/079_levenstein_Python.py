def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    
    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    
    return dp[m][n]


if __name__ == "__main__":
    import sys
    s1 = sys.stdin.readline().strip()
    s2 = sys.stdin.readline().strip()
    print(levenshtein_distance(s1, s2))

