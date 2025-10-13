'===============================================================================
' BFS Shortest Path on an Unweighted Graph (VB.NET, single-file console program)
'-------------------------------------------------------------------------------
' INPUT (from STDIN; whitespace/tab separated):
'   • Edge line : U  V1 [V2 ...]   -> add UNDIRECTED edges U—V1, U—V2, ...
'   • Node line : U                -> ensure node exists (even if isolated)
'   • Query line: #  SRC  DST     -> request shortest path from SRC to DST
'
' Example:
'   A   B   F
'   B   A   C
'   C   B   D
'   D   C   E
'   E   D   F
'   F   A   E
'   #   A   E
'
' RUN (example):
'   dotnet run < graph.tsv
'
' OUTPUT:
'   For each "# SRC DST" query, prints either "A -> B -> ... -> E"
'   or "No path found from SRC to DST".
'
' NOTES:
'   • Uses a Dictionary(string→index) and lists to store labels and adjacency.
'   • Adjacency is kept as HashSet(Of Integer) to avoid parallel edges.
'   • BFS is O(|V|+|E|) per query; memory is linear in graph size.
'===============================================================================

Imports System
Imports System.Collections.Generic
Imports System.Text.RegularExpressions

Module Program
    Sub Main()
        Dim graph As New Graph()
        Dim queries As New List(Of Tuple(Of String, String))()

        '------------------------------- Parse input ----------------------------
        While True
            Dim line As String = Console.ReadLine()
            If line Is Nothing Then Exit While

            line = line.Trim()
            If line.Length = 0 Then Continue While

            Dim tokens As String() = Regex.Split(line, "\s+")
            If tokens.Length = 0 Then Continue While

            If tokens(0) = "#"c OrElse tokens(0) = "#" Then
                ' Query line: "# SRC DST"
                If tokens.Length >= 3 Then
                    Dim s As String = tokens(1).Trim()
                    Dim d As String = tokens(2).Trim()
                    ' Keep labels known (harmless even if they weren't in any edge)
                    graph.GetOrAddIndex(s)
                    graph.GetOrAddIndex(d)
                    queries.Add(Tuple.Create(s, d))
                End If
            Else
                ' Edge or isolated-node line: "U [V1 V2 ...]"
                Dim uLabel As String = tokens(0).Trim()
                Dim u As Integer = graph.GetOrAddIndex(uLabel)

                If tokens.Length > 1 Then
                    For i As Integer = 1 To tokens.Length - 1
                        Dim vLabel As String = tokens(i).Trim()
                        If vLabel.Length = 0 Then Continue For
                        Dim v As Integer = graph.GetOrAddIndex(vLabel)
                        graph.AddUndirected(u, v)
                    Next
                End If
            End If
        End While

        '------------------------------ Answer queries --------------------------
        For Each q In queries
            Dim path As List(Of String) = graph.BfsPathLabels(q.Item1, q.Item2)
            If path Is Nothing OrElse path.Count = 0 Then
                Console.WriteLine("No path found from {0} to {1}", q.Item1, q.Item2)
            Else
                Console.WriteLine(String.Join(" -> ", path))
            End If
        Next
    End Sub
End Module

'-------------------------------- Graph type -----------------------------------
Friend Class Graph
    Private ReadOnly labelToIndex As Dictionary(Of String, Integer)
    Private ReadOnly indexToLabel As List(Of String)
    Private ReadOnly adj As List(Of HashSet(Of Integer))

    Public Sub New()
        labelToIndex = New Dictionary(Of String, Integer)(StringComparer.Ordinal)
        indexToLabel = New List(Of String)()
        adj = New List(Of HashSet(Of Integer))()
    End Sub

    ' Add label if new; return its 0-based index.
    Public Function GetOrAddIndex(label As String) As Integer
        label = label.Trim()
        Dim idx As Integer
        If labelToIndex.TryGetValue(label, idx) Then
            Return idx
        End If
        idx = indexToLabel.Count
        labelToIndex(label) = idx
        indexToLabel.Add(label)
        adj.Add(New HashSet(Of Integer)())
        Return idx
    End Function

    ' Find existing label; return -1 if absent.
    Public Function FindIndex(label As String) As Integer
        label = label.Trim()
        Dim idx As Integer
        If labelToIndex.TryGetValue(label, idx) Then
            Return idx
        End If
        Return -1
    End Function

    ' Insert an undirected edge (u—v), ignoring self-loops and duplicates.
    Public Sub AddUndirected(u As Integer, v As Integer)
        If u = v Then Return
        adj(u).Add(v)
        adj(v).Add(u)
    End Sub

    ' BFS shortest path between two labels; returns list of labels or Nothing.
    Public Function BfsPathLabels(srcLabel As String, dstLabel As String) As List(Of String)
        Dim s As Integer = FindIndex(srcLabel)
        Dim d As Integer = FindIndex(dstLabel)

        If s = -1 OrElse d = -1 Then
            Return Nothing
        End If
        If s = d Then
            Return New List(Of String)() From {indexToLabel(s)}
        End If

        Dim n As Integer = indexToLabel.Count
        Dim parent(n - 1) As Integer
        For i As Integer = 0 To n - 1 : parent(i) = -1 : Next
        Dim vis(n - 1) As Boolean

        Dim q As New Queue(Of Integer)()
        vis(s) = True
        q.Enqueue(s)

        Dim found As Boolean = False
        While q.Count > 0 AndAlso Not found
            Dim u As Integer = q.Dequeue()
            For Each v As Integer In adj(u)
                If Not vis(v) Then
                    vis(v) = True
                    parent(v) = u
                    If v = d Then
                        found = True
                        Exit For
                    End If
                    q.Enqueue(v)
                End If
            Next
        End While

        If Not vis(d) Then
            Return Nothing
        End If

        ' Reconstruct path d -> ... -> s, then reverse.
        Dim stack As New List(Of Integer)()
        Dim cur As Integer = d
        While cur <> -1
            stack.Add(cur)
            cur = parent(cur)
        End While
        stack.Reverse()

        Dim labels As New List(Of String)(stack.Count)
        For Each idx In stack
            labels.Add(indexToLabel(idx))
        Next
        Return labels
    End Function
End Class

