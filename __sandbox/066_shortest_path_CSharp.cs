// Program.cs
// Breadth-First Search (BFS) shortest path on an unweighted, *undirected* graph.
//
// INPUT FORMAT (tab- or space-separated; one record per line):
//   Edge lines: "U  V1 [V2 ...]"   => add undirected edges U—V1, U—V2, ...
//   Query lines: "#  SRC  DST"     => request shortest path from SRC to DST
//
// Example (matches your prompt):
//   A   B   F
//   B   A   C
//   C   B   D
//   D   C   E
//   E   D   F
//   F   A   E
//   #   A   E
//
// USAGE
//   dotnet run --project . < graph.tsv
//   # or: dotnet run --project . path/to/graph.tsv

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

internal static class Program
{
    private static void Main(string[] args)
    {
        using var reader = args.Length > 0
            ? new StreamReader(args[0])
            : new StreamReader(Console.OpenStandardInput());

        var (adj, queries) = ParseInput(reader);

        if (queries.Count == 0)
        {
            Console.WriteLine("No queries found (expect lines like: \"# SRC DST\").");
            return;
        }

        foreach (var (src, dst) in queries)
        {
            if (src == dst)
            {
                Console.WriteLine(src);
                continue;
            }

            if (!adj.ContainsKey(src))
            {
                Console.WriteLine($"No path found from {src} to {dst} (source not in graph)");
                continue;
            }
            if (!adj.ContainsKey(dst))
            {
                Console.WriteLine($"No path found from {src} to {dst} (destination not in graph)");
                continue;
            }

            var path = BfsPath(adj, src, dst);
            if (path is null)
                Console.WriteLine($"No path found from {src} to {dst}");
            else
                Console.WriteLine(string.Join(" -> ", path));
        }
    }

    // ------------------------------- Parsing -------------------------------

    private static (Dictionary<string, List<string>> adj, List<(string src, string dst)> queries)
        ParseInput(TextReader reader)
    {
        var adj = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var queries = new List<(string src, string dst)>();

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            line = line.Trim();
            if (line.Length == 0) continue;

            var tok = SplitWs(line);
            if (tok.Count == 0) continue;

            if (tok[0].StartsWith("#"))
            {
                if (tok.Count >= 3)
                    queries.Add((tok[1], tok[2]));
                continue;
            }

            var u = tok[0];
            if (tok.Count == 1)
            {
                EnsureKey(adj, u);
                continue;
            }

            for (int i = 1; i < tok.Count; i++)
                AddUndirectedEdge(adj, u, tok[i]);
        }

        return (adj, queries);
    }

    private static List<string> SplitWs(string s)
    {
        var list = new List<string>();
        int i = 0, n = s.Length;
        while (i < n)
        {
            while (i < n && char.IsWhiteSpace(s[i])) i++;
            if (i >= n) break;
            int j = i + 1;
            while (j < n && !char.IsWhiteSpace(s[j])) j++;
            list.Add(s[i..j]);
            i = j + 1;
        }
        return list;
    }

    private static void AddUndirectedEdge(Dictionary<string, List<string>> adj, string u, string v)
    {
        AddDirectedUnique(adj, u, v);
        AddDirectedUnique(adj, v, u);
        EnsureKey(adj, u);
        EnsureKey(adj, v);
    }

    private static void AddDirectedUnique(Dictionary<string, List<string>> adj, string u, string v)
    {
        if (!adj.TryGetValue(u, out var list))
        {
            list = new List<string>();
            adj[u] = list;
        }
        // Avoid duplicates (keeps memory stable on repeated edge declarations)
        if (!list.Contains(v)) list.Add(v);
    }

    private static void EnsureKey(Dictionary<string, List<string>> adj, string u)
    {
        if (!adj.ContainsKey(u)) adj[u] = new List<string>();
    }

    // ------------------------------- BFS -----------------------------------

    // Returns shortest path [src, ..., dst] or null if unreachable.
    private static List<string>? BfsPath(Dictionary<string, List<string>> adj, string src, string dst)
    {
        var visited = new HashSet<string>(StringComparer.Ordinal) { src };
        var parent = new Dictionary<string, string>(StringComparer.Ordinal);
        var q = new Queue<string>();
        q.Enqueue(src);

        while (q.Count > 0)
        {
            var u = q.Dequeue();
            if (!adj.TryGetValue(u, out var nbrs)) continue;

            // Optional: stable neighbor order for reproducible paths
            foreach (var v in nbrs)
            {
                if (visited.Contains(v)) continue;
                visited.Add(v);
                parent[v] = u;
                if (v == dst)
                    return Reconstruct(parent, src, dst);
                q.Enqueue(v);
            }
        }
        return null;
    }

    private static List<string> Reconstruct(Dictionary<string, string> parent, string src, string dst)
    {
        var path = new List<string>();
        var cur = dst;
        path.Add(cur);
        while (cur != src)
        {
            if (!parent.TryGetValue(cur, out var p))
                break; // safety: malformed call
            cur = p;
            path.Add(cur);
        }
        path.Reverse();
        return path;
    }
}

