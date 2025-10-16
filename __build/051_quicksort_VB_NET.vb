' In-place quicksort on an integer array (reads all integers from STDIN)
' Build & run (Windows or cross-platform with .NET SDK):
'   dotnet new console -n QsApp -lang "VB"
'   Replace Program.vb with this file (or paste into it)
'   dotnet run --project QsApp < input.txt
' Or compile with vbc if available.

Imports System
Imports System.Collections.Generic

Module Program
    Sub Main()
        ' ---- 1) Read integers from STDIN (any whitespace-separated format) ----
        Dim raw As String = Console.In.ReadToEnd()
        Dim seps As Char() = {ControlChars.Cr, ControlChars.Lf, ControlChars.Tab, " "c}
        Dim tokens As String() = raw.Split(seps, StringSplitOptions.RemoveEmptyEntries)

        Dim data As New List(Of Integer)(Math.Max(16, tokens.Length))
        For Each t In tokens
            Dim v As Integer
            If Integer.TryParse(t, v) Then
                data.Add(v)
            End If
        Next

        If data.Count = 0 Then
            Return ' nothing to do
        End If

        Dim a As Integer() = data.ToArray()

        ' ---- 2) In-place quicksort (Hoare partition, median-of-three pivot) ----
        QuickSort(a, 0, a.Length - 1)

        ' ---- 3) Emit result ----
        For Each v In a
            Console.Out.WriteLine(v)
        Next
    End Sub

    Private Sub QuickSort(ByRef arr As Integer(), ByVal lo As Integer, ByVal hi As Integer)
        If lo >= hi Then Return

        Dim i As Integer = lo
        Dim j As Integer = hi
        Dim mid As Integer = lo + (hi - lo) \ 2
        Dim pivot As Integer = MedianOfThree(arr(lo), arr(mid), arr(hi))

        Do
            While arr(i) < pivot
                i += 1
            End While
            While arr(j) > pivot
                j -= 1
            End While

            If i <= j Then
                Swap(arr(i), arr(j))
                i += 1
                j -= 1
            End If
        Loop While i <= j

        If lo < j Then QuickSort(arr, lo, j)
        If i < hi Then QuickSort(arr, i, hi)
    End Sub

    Private Function MedianOfThree(ByVal a As Integer, ByVal b As Integer, ByVal c As Integer) As Integer
        If (a >= b AndAlso a <= c) OrElse (a <= b AndAlso a >= c) Then
            Return a
        ElseIf (b >= a AndAlso b <= c) OrElse (b <= a AndAlso b >= c) Then
            Return b
        Else
            Return c
        End If
    End Function

    Private Sub Swap(ByRef x As Integer, ByRef y As Integer)
        If x = y Then Return
        Dim t As Integer = x
        x = y
        y = t
    End Sub
End Module

