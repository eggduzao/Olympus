using System;

class LevenshteinDistance
{
    static int Compute(string s1, string s2)
    {
        int m = s1.Length;
        int n = s2.Length;
        int[,] dp = new int[m + 1, n + 1];

        for (int i = 0; i <= m; i++)
            dp[i, 0] = i;
        for (int j = 0; j <= n; j++)
            dp[0, j] = j;

        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
                int del = dp[i - 1, j] + 1;
                int ins = dp[i, j - 1] + 1;
                int sub = dp[i - 1, j - 1] + cost;
                dp[i, j] = Math.Min(Math.Min(del, ins), sub);
            }
        }

        return dp[m, n];
    }

    static void Main()
    {
        string s1 = Console.ReadLine();
        string s2 = Console.ReadLine();

        int distance = Compute(s1, s2);
        Console.WriteLine(distance);
    }
}

