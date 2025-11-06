' Program.vb — Binary search (lower-bound) over a sorted ascending list.
' Build (Windows or Linux/OSX with .NET SDK):
'   dotnet new console -n BinSearchVB -f net8.0
'   mv Program.vb BinSearchVB/Program.vb
'   cd BinSearchVB && dotnet run -- <path_or_->_ <target_int>
'
' Usage:
'   dotnet run -- <path_or_->_ <target_int>
'     <path_or_->_ : path to a text file with one integer per line; use "-" for STDIN
'     <target_int> : integer to search for (Int64)
'
' Behavior:
'   • Ignores blank lines and lines that can’t be parsed as integers (prints a warning count).
'   • Verifies ascending (non-decreasing) order before searching.
'   • Performs lower-bound binary search (first index with a(i) >= target), 1-based reporting.

Imports System
Imports System.IO
Imports System.Collections.Generic

Module Program

  Sub Main(args As String())
    If args.Length < 2 Then
      Console.WriteLine("Usage: dotnet run -- <path_or_->_ <target_int>")
      Console.WriteLine("Example:")
      Console.WriteLine("  printf ""1\n3\n4\n7\n9\n11\n15\n"" | dotnet run -- - 11")
      Environment.
    End If

    Dim src As String = args(0).Trim()
    Dim target As Long
    If Not Int64.TryParse(args(1), target) Then
      Console.Error.WriteLine($"ERROR: TARGET must be an integer, got: {args(1)}")
      Environment.
    End If

    Dim data As New List(Of Long)(capacity:=1024)
    Dim skipped As Integer = 0

    If src = "-"c Then
      ReadStream(Console.In, data, skipped)
    Else
      Try
        Using sr As New StreamReader(src)
          ReadStream(sr, data, skipped)
        End Using
      Catch ex As Exception
        Console.Error.WriteLine($"ERROR: Cannot open file: {src} ({ex.Message})")
        Environment.
      End Try
    End If

    If skipped > 0 Then
      Console.WriteLine($"WARN: skipped {skipped} non-integer line(s).")
    End If
    If data.Count = 0 Then
      Console.Error.WriteLine("ERROR: No numeric input found.")
      Environment.
    End If

    Dim badPair As (Long, Long) = CheckSorted(data)
    If badPair.Item1 <> 0 OrElse badPair.Item2 <> 0 Then
      Console.Error.WriteLine($"ERROR: Input not in ascending order near {badPair.Item1} then {badPair.Item2}")
      Environment.
    End If

    Dim pos As Integer = LowerBound(data, target) ' 0-based position where a(pos) >= target
    If pos < data.Count AndAlso data(pos) = target Then
      Console.WriteLine($"FOUND {target} at index {pos + 1}")
    Else
      Dim leftVal As String = If(pos > 0, data(pos - 1).ToString(), "-inf")
      Dim rightVal As String = If(pos < data.Count, data(pos).ToString(), "+inf")
      Console.WriteLine($"NOT FOUND {target}. Insertion index {pos + 1} (1-based), between {leftVal} and {rightVal}")
    End If
  End Sub

  ' Read integers from a TextReader, appending into list; count non-integer lines.
  Private Sub ReadStream(r As TextReader, ByRef acc As List(Of Long), ByRef skipped As Integer)
    Dim line As String
    Do
      line = r.ReadLine()
      If line Is Nothing Then Exit Do
      Dim s As String = line.Trim()
      If s.Length = 0 Then Continue Do
      Dim v As Long
      If Int64.TryParse(s, v) Then
        acc.Add(v)
      Else
        skipped += 1
      End If
    Loop
  End Sub

  ' Return (0,0) if sorted non-decreasing; otherwise the offending adjacent pair (prev, curr).
  Private Function CheckSorted(a As List(Of Long)) As (Long, Long)
    For i As Integer = 1 To a.Count - 1
      If a(i) < a(i - 1) Then
        Return (a(i - 1), a(i))
      End If
    Next
    Return (0L, 0L)
  End Function

  ' LowerBound: first index i in [0..n] with a(i) >= target (if none, returns n).
  Private Function LowerBound(a As List(Of Long), target As Long) As Integer
    Dim lo As Integer = 0
    Dim hi As Integer = a.Count
    While lo < hi
      Dim mid As Integer = lo + (hi - lo) \ 2
      If a(mid) < target Then
        lo = mid + 1
      Else
        hi = mid
      End If
    End While
    Return lo
  End Function

End Module

