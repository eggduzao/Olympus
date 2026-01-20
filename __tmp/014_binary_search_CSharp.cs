// Program.cs
// C# CLI: Binary search (first occurrence) on a sorted ascending array of integers.
// Usage:
//   dotnet run -- <target> [path/to/file]
//   # or from STDIN (one number per line):
//   printf "1\n3\n4\n7\n9\n11\n15\n" | dotnet run -- 11
//
// Behavior
//   • O(log N) binary search, returns the FIRST index if duplicates exist (stable-left).
//   • On hit : prints  FOUND <target> at index <i> (1-based)
//   • On miss: prints  NOT FOUND <target>. Insertion index <i> (1-based), between <left> and <right>
//   • Validates non-decreasing order; exits with error if violated.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

static class Program
{
    static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("ERROR: missing <target>");
            Usage();
            return 2;
        }

        if (!long.TryParse(args[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var target))
        {
            Console.Error.WriteLine($"ERROR: target must be integer, got '{args[0]}'");
            Usage();
            return 2;
        }

        List<long> numbers;
        try
        {
            if (args.Length >= 2)
            {
                var path = args[1];
                using var sr = new StreamReader(path);
                numbers = ReadNumbers(sr);
            }
            else
            {
                using var sr = new StreamReader(Console.OpenStandardInput());
                numbers = ReadNumbers(sr);
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"ERROR: failed to read input: {ex.Message}");
            return 1;
        }

        if (numbers.Count == 0)
        {
            Console.Error.WriteLine("ERROR: no numeric input provided.");
            return 1;
        }

        try { EnsureAscending(numbers); }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }

        var (idx1, found, ins1) = BinarySearchFirst(numbers, target);

        if (found)
        {
            Console.WriteLine($"FOUND {target} at index {idx1} (1-based)");
        }
        else
        {
            var left  = ins1 >= 2 ? numbers[ins1 - 2].ToString(CultureInfo.InvariantCulture) : "-inf";
            var right = (ins1 - 1) < numbers.Count ? numbers[ins1 - 1].ToString(CultureInfo.InvariantCulture) : "+inf";
            Console.WriteLine($"NOT FOUND {target}. Insertion index {ins1} (1-based), between {left} and {right}");
        }

        return 0;
    }

    // Read integers (one per line). Non-numeric lines are skipped with a warning.
    static List<long> ReadNumbers(TextReader tr)
    {
        var list = new List<long>();
        string? line;
        int lineNo = 0;
        while ((line = tr.ReadLine()) != null)
        {
            lineNo++;
            var s = line.Trim();
            if (s.Length == 0) continue;

            if (long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v))
            {
                list.Add(v);
            }
            else
            {
                Console.Error.WriteLine($"WARN: skipping non-integer line {lineNo}: {s}");
            }
        }
        return list;
    }

    // Ensure non-decreasing order.
    static void EnsureAscending(IReadOnlyList<long> a)
    {
        for (int i = 1; i < a.Count; i++)
        {
            if (a[i] < a[i - 1])
                throw new InvalidOperationException(
                    $"ERROR: input not in ascending order at position {i + 1}: {a[i]} < {a[i - 1]}");
        }
    }

    // Binary search for first occurrence (stable-left).
    // Returns: (index_1based, found_bool, insertion_index_1based)
    static (int idx1, bool found, int ins1) BinarySearchFirst(IReadOnlyList<long> a, long target)
    {
        int lo = 0, hi = a.Count - 1;
        int found = -1;

        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            long v = a[mid];
            if (v == target)
            {
                found = mid;
                hi = mid - 1; // continue left
            }
            else if (v < target)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        if (found >= 0) return (found + 1, true, lo + 1);
        return (0, false, lo + 1);
    }

    static void Usage()
    {
        Console.Error.WriteLine("Usage:");
        Console.Error.WriteLine("  dotnet run -- <target> [path/to/file]");
        Console.Error.WriteLine("  printf \"1\\n3\\n4\\n7\\n9\\n11\\n15\\n\" | dotnet run -- 11");
    }
}
