#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int min(int a, int b, int c) {
    if (a <= b && a <= c) return a;
    if (b <= a && b <= c) return b;
    return c;
}

int levenshtein(const char *s1, const char *s2) {
    int m = strlen(s1);
    int n = strlen(s2);

    int **dp = malloc((m + 1) * sizeof(int *));
    for (int i = 0; i <= m; i++) {
        dp[i] = malloc((n + 1) * sizeof(int));
    }

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            dp[i][j] = min(
                dp[i - 1][j] + 1,     // Deletion
                dp[i][j - 1] + 1,     // Insertion
                dp[i - 1][j - 1] + cost // Substitution
            );
        }
    }

    int result = dp[m][n];

    for (int i = 0; i <= m; i++) free(dp[i]);
    free(dp);

    return result;
}

int main() {
    char s1[1024], s2[1024];
    if (fgets(s1, sizeof(s1), stdin) == NULL) return 1;
    if (fgets(s2, sizeof(s2), stdin) == NULL) return 1;

    // Remove trailing newlines
    s1[strcspn(s1, "\n")] = 0;
    s2[strcspn(s2, "\n")] = 0;

    int dist = levenshtein(s1, s2);
    printf("%d\n", dist);
    return 0;
}

