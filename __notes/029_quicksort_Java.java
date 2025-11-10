import java.io.*;
import java.util.*;

/**
 * In-place quicksort on an array (iterative, Hoare partition, median-of-three).
 *
 * Usage
 * -----
 * # stdin (one number per line)
 *   javac QuicksortInPlace.java && java QuicksortInPlace < numbers.txt
 *
 * # or pass a file path
 *   javac QuicksortInPlace.java && java QuicksortInPlace numbers.txt
 *
 * Input format
 * ------------
 * - One number per line (integer or float).
 * - Blank lines are ignored.
 * - Lines beginning with '#' are comments and ignored.
 *
 * Output
 * ------
 * - Sorted numbers, one per line.
 *   (Integers print without a decimal; non-integers print as-is.)
 */
public final class QuicksortInPlace {

    // ---------- I/O ----------
    private static double[] readNumbers(Reader rdr) throws IOException {
        BufferedReader br = new BufferedReader(rdr);
        ArrayList<Double> buf = new ArrayList<>(1 << 12);
        String line;
        int lineNo = 0;
        while ((line = br.readLine()) != null) {
            lineNo++;
            String s = line.trim();
            if (s.isEmpty() || s.startsWith("#")) continue;
            try {
                // Accept ints or floats; everything as double for simplicity
                double v = Double.parseDouble(s);
                if (!Double.isFinite(v)) {
                    throw new NumberFormatException("non-finite");
                }
                buf.add(v);
            } catch (NumberFormatException nfe) {
                throw new IOException("Invalid numeric line at " + lineNo + ": " + s);
            }
        }
        double[] out = new double[buf.size()];
        for (int i = 0; i < buf.size(); i++) out[i] = buf.get(i);
        return out;
    }

    private static void printNumbers(double[] a, Appendable out) throws IOException {
        for (double v : a) {
            // Pretty-print integers without trailing ".0"
            if (v == Math.rint(v)) {
                long asLong = (long) Math.rint(v);
                out.append(Long.toString(asLong)).append('\n');
            } else {
                out.append(Double.toString(v)).append('\n');
            }
        }
    }

    // ---------- Sorting primitives ----------
    private static int medianOfThree(double[] a, int i, int j, int k) {
        double ai = a[i], aj = a[j], ak = a[k];
        if (ai < aj) {
            if (aj < ak) return j;       // ai < aj < ak
            return (ai < ak) ? i : k;    // ai < ak <= aj  OR  ak <= ai < aj
        } else {
            if (ai < ak) return i;       // aj <= ai < ak
            return (aj < ak) ? j : k;    // aj < ak <= ai OR ak <= aj <= ai
        }
    }

    private static void insertionSort(double[] a, int lo, int hi) {
        for (int i = lo + 1; i <= hi; i++) {
            double key = a[i];
            int j = i - 1;
            while (j >= lo && a[j] > key) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = key;
        }
    }

    /**
     * Hoare partitioning.
     * Returns index p such that range [lo..p] <= pivot and [p+1..hi] >= pivot (not stable).
     */
    private static int hoarePartition(double[] a, int lo, int hi) {
        int mid = lo + ((hi - lo) >>> 1);
        int pidx = medianOfThree(a, lo, mid, hi);
        double pivot = a[pidx];

        int i = lo - 1;
        int j = hi + 1;
        while (true) {
            do { i++; } while (a[i] < pivot);
            do { j--; } while (a[j] > pivot);
            if (i >= j) return j;
            double tmp = a[i]; a[i] = a[j]; a[j] = tmp;
        }
    }

    /**
     * Iterative quicksort with small-range cutoff to insertion sort.
     */
    public static void quicksortInPlace(double[] a) {
        final int n = a.length;
        if (n < 2) return;

        final int SMALL = 24; // cutoff for insertion sort (tuneable)
        Deque<int[]> stack = new ArrayDeque<>();
        stack.push(new int[]{0, n - 1});

        while (!stack.isEmpty()) {
            int[] range = stack.pop();
            int lo = range[0], hi = range[1];
            if (hi - lo + 1 <= SMALL) {
                insertionSort(a, lo, hi);
                continue;
            }

            int p = hoarePartition(a, lo, hi);
            int leftSize  = p - lo + 1;
            int rightSize = hi - (p + 1) + 1;

            // Tail-recursion elimination style: push larger side first, process smaller next
            if (leftSize < rightSize) {
                if (p + 1 < hi) stack.push(new int[]{p + 1, hi});
                if (lo < p)     stack.push(new int[]{lo, p});
            } else {
                if (lo < p)     stack.push(new int[]{lo, p});
                if (p + 1 < hi) stack.push(new int[]{p + 1, hi});
            }
        }
    }

    // ---------- Main ----------
    public static void main(String[] args) {
        try {
            double[] arr;
            if (args.length >= 1) {
                try (FileReader fr = new FileReader(args[0])) {
                    arr = readNumbers(fr);
                }
            } else {
                InputStreamReader isr = new InputStreamReader(System.in);
                arr = readNumbers(isr);
            }

            quicksortInPlace(arr);
            printNumbers(arr, new OutputStreamWriter(System.out));
        } catch (IOException ioe) {
            System.err.println(ioe.getMessage());
            System.exit(1);
        }
    }
}

