' Levenshtein edit distance in VB.NET
' Reads two strings from console input and prints the distance.

Module Levenshtein
    Function ComputeDistance(ByVal s As String, ByVal t As String) As Integer
        Dim m As Integer = s.Length
        Dim n As Integer = t.Length
        Dim dp(m, n) As Integer

        ' Base cases
        For i As Integer = 0 To m
            dp(i, 0) = i
        Next
        For j As Integer = 0 To n
            dp(0, j) = j
        Next

        ' Fill DP table
        For i As Integer = 1 To m
            For j As Integer = 1 To n
                Dim cost As Integer = If(s(i - 1) = t(j - 1), 0, 1)
                dp(i, j) = Math.Min(
                    Math.Min(dp(i - 1, j) + 1, dp(i, j - 1) + 1),
                    dp(i - 1, j - 1) + cost
                )
            Next
        Next

        Return dp(m, n)
    End Function

    Sub Main()
        Dim s1 As String = Console.ReadLine()
        Dim s2 As String = Console.ReadLine()
        Dim distance As Integer = ComputeDistance(s1, s2)
        Console.WriteLine(distance)
    End Sub
End Module

