/**
 * Binary search in a sorted array — Java (single-file, CLI)
 *
 * Usage:
 *   javac BinarySearch.java
 *   java BinarySearch input.txt 1 9 15 2
 *
 *   - input.txt: text file with one integer per line (unsorted OK; we sort)
 *   - following args are targets to query (if none are given, uses a demo set)
 *
 * Notes:
 *   • Time: O(log n) per query after O(n log n) sort.
 *   • Space: O(n) to hold the array (random access is needed for binary search).
 *   • For multi-GB inputs that don't fit in RAM, you’d need an external index
 *     or on-disk sorted structure (e.g., SQLite + index) instead of plain arrays.
 */

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BinarySearch {

    /**
     * Binary search over a sorted long[].
     * @param arr   sorted ascending
     * @param target value to find
     * @return index of the target if found; otherwise -1
     */
    public static int binarySearch(long[] arr, long target) {
        int lo = 0, hi = arr.length - 1;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >>> 1); // avoid overflow
            long v = arr[mid];
            if (v == target) return mid;
            if (v < target)  lo = mid + 1;
            else             hi = mid - 1;
        }
        return -1;
    }

    /**
     * Load integers from a text file (one per line), skipping malformed lines.
     * Returns a sorted long[] (ascending).
     */
    public static long[] loadArrayFromFile(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new IOException("Input file not found: " + path);
        }

        List<Long> buf = new ArrayList<>(1 << 16); // start with 65,536 capacity
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            int lineNo = 0;
            while ((line = br.readLine()) != null) {
                lineNo++;
                line = line.trim();
                if (line.isEmpty()) continue;
                try {
                    // Accepts plain integers (fits in signed 64-bit)
                    long v = Long.parseLong(line);
                    buf.add(v);
                } catch (NumberFormatException nfe) {
                    // Non-fatal: skip malformed lines
                    System.err.println("[skip] line " + lineNo + " not a valid integer: " + line);
                }
            }
        }

        long[] arr = new long[buf.size()];
        for (int i = 0; i < buf.size(); i++) arr[i] = buf.get(i);
        Arrays.sort(arr);
        return arr;
    }

    private static void report(long target, int idx) {
        if (idx >= 0) {
            System.out.println("Target " + target + ": found at index " + idx);
        } else {
            System.out.println("Target " + target + ": not found");
        }
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java BinarySearch <input.txt> [targets ...]");
            System.err.println("Example: java BinarySearch input.txt 1 9 15 2");
            System.exit(2);
        }

        Path input = Path.of(args[0]);
        long[] arr;
        try {
            arr = loadArrayFromFile(input);
        } catch (IOException e) {
            System.err.println("Failed to read input: " + e.getMessage());
            System.exit(1);
            return;
        }
        System.out.println("Loaded " + arr.length + " integers from " + input);

        // Parse targets (if any). If none, use a small demo set.
        List<Long> targets = new ArrayList<>();
        if (args.length > 1) {
            for (int i = 1; i < args.length; i++) {
                try {
                    targets.add(Long.parseLong(args[i]));
                } catch (NumberFormatException nfe) {
                    System.err.println("[skip target] not an integer: " + args[i]);
                }
            }
        }
        if (targets.isEmpty()) {
            targets = Arrays.asList(1L, 9L, 15L, 2L);
        }

        for (long t : targets) {
            int idx = binarySearch(arr, t);
            report(t, idx);
        }
    }
}
