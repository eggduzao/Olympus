// QuicksortInPlace.cs
// In-place quicksort on an array in C#.
//
// • Reads integers (one per line) from a file path passed as the first CLI arg,
//   or from STDIN if no file is given.
// • Skips blank lines and lines starting with '#'.
// • Sorts in place using an iterative quicksort with Hoare partition
//   and a median-of-three pivot to reduce worst-case behavior.
// • Prints the sorted numbers, one per line, to STDOUT.
//
// Build & run:
//   csc QuicksortInPlace.cs && ./QuicksortInPlace input.txt
//   # or
//   printf "5\n3\n8\n1\n2\n" | ./QuicksortInPlace

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

public static class QuicksortInPlace
{
    public static void Main(string[] args)
    {
        try
        {
            var nums = args.Length > 0 ? ReadNumbers(args[0]) : ReadNumbers();
            if (nums.Count <= 1)
            {
                foreach (var v in nums) Console.WriteLine(v);
                return;
            }

            // Sort in place
            var arr = nums.ToArray();
            QuickSort(arr);

            // Output
            foreach (var v in arr) Console.WriteLine(v);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            Environment.ExitCode = 1;
        }
    }

    // ---------- IO ----------
    private static List<long> ReadNumbers(string path)
    {
        using var sr = new StreamReader(path);
        return ReadNumbersCore(sr);
    }

    private static List<long> ReadNumbers()
    {
        using var sr = new StreamReader(Console.OpenStandardInput());
        return ReadNumbersCore(sr);
    }

    private static List<long> ReadNumbersCore(StreamReader sr)
    {
        var list = new List<long>(capacity: 1 << 12);
        string? line;
        int lineno = 0;

        while ((line = sr.ReadLine()) is not null)
        {
            lineno++;
            line = line.Trim();
            if (line.Length == 0 || line.StartsWith("#")) continue;

            if (!long.TryParse(line, NumberStyles.Integer, CultureInfo.InvariantCulture, out var val))
                throw new FormatException($"line {lineno}: not an integer: {line}");

            list.Add(val);
        }
        return list;
    }

    // ---------- Quicksort (iterative, Hoare partition, median-of-three pivot) ----------
    public static void QuickSort(long[] a)
    {
        if (a.Length < 2) return;

        var stack = new Stack<(int lo, int hi)>();
        stack.Push((0, a.Length - 1));

        while (stack.Count > 0)
        {
            var (lo, hi) = stack.Pop();
            if (lo >= hi) continue;

            // Partition and get split point
            int p = HoarePartition(a, lo, hi);

            // Process smaller side first to keep stack shallow
            int leftSize = p - lo + 1;      // [lo..p]
            int rightSize = hi - (p + 1) + 1; // [p+1..hi]
            if (leftSize < rightSize)
            {
                if (p + 1 < hi) stack.Push((p + 1, hi));
                if (lo < p)     stack.Push((lo, p));
            }
            else
            {
                if (lo < p)     stack.Push((lo, p));
                if (p + 1 < hi) stack.Push((p + 1, hi));
            }
        }
    }

    // Hoare partition using median-of-three pivot selection.
    private static int HoarePartition(long[] a, int lo, int hi)
    {
        int mid = lo + ((hi - lo) >> 1);
        long pivot = MedianOfThree(a[lo], a[mid], a[hi]);

        int i = lo - 1;
        int j = hi + 1;
        while (true)
        {
            do { i++; } while (a[i] < pivot);
            do { j--; } while (a[j] > pivot);
            if (i >= j) return j;
            (a[i], a[j]) = (a[j], a[i]);
        }
    }

    private static long MedianOfThree(long x, long y, long z)
    {
        // Return the median value among x, y, z without allocations.
        if (x > y) (x, y) = (y, x);
        if (y > z) (y, z) = (z, y);
        if (x > y) (x, y) = (y, x);
        return y; // now y is the median
    }
}

