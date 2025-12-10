// binary_search_cli.go
//
// Binary search over a sorted (ascending) list of numbers.
// - Works as a CLI:  go run binary_search_cli.go <target> [path/to/file]
//   • If <file> is omitted, the program reads numbers from STDIN (one per line).
// - Works as a library-like function too: see binarySearch() below.
//
// Behavior
//   • O(log N) classic binary search.
//   • Returns FIRST occurrence if duplicates exist (stable-left).
//   • On hit:  prints  FOUND <target> at index <i> (1-based)
//   • On miss: prints  NOT FOUND <target>. Insertion index <i> (1-based), between <left> and <right>
//
// Input requirements
//   • Non-decreasing order (ascending, duplicates allowed). Program validates and errors out if violated.
//
// Build & Run examples
//   go run binary_search_cli.go 11 data.txt
//   printf "1\n3\n4\n7\n9\n11\n15\n" | go run binary_search_cli.go 11
//   go build -o bsearch && ./bsearch 5 data.txt
//
// Notes
//   • Uses float64 for generality. Change to int if desired.

package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// -----------------------------
// Core algorithm (library API)
// -----------------------------

// binarySearch performs binary search over ascending slice `arr` for `target`.
// Returns (idx, found, insertPos) where:
//   - idx       : 1-based index if found, else 0
//   - found     : true if an exact match exists
//   - insertPos : 1-based position where `target` should be inserted to keep order (always valid)
func binarySearch(arr []float64, target float64) (int, bool, int) {
	lo, hi := 0, len(arr)-1
	foundIdx := -1

	for lo <= hi {
		mid := lo + (hi-lo)/2
		v := arr[mid]
		switch {
		case v == target:
			foundIdx = mid
			// keep searching left for the first occurrence
			hi = mid - 1
		case v < target:
			lo = mid + 1
		default:
			hi = mid - 1
		}
	}

	if foundIdx >= 0 {
		return foundIdx + 1, true, lo + 1 // insertPos still lo+1
	}
	return 0, false, lo + 1
}

// ensureAscending validates non-decreasing order (ascending) of `arr`.
func ensureAscending(arr []float64) error {
	if len(arr) <= 1 {
		return nil
	}
	last := arr[0]
	for i := 1; i < len(arr); i++ {
		if arr[i] < last {
			return fmt.Errorf("input data is not in ascending order at position %d: %.10g < %.10g", i+1, arr[i], last)
		}
		last = arr[i]
	}
	return nil
}

// -----------------------------
// I/O helpers
// -----------------------------

// readNumbers reads float64s from reader, one per line (blank lines ignored).
// Non-numeric lines are skipped with a warning to STDERR.
func readNumbers(r io.Reader) ([]float64, error) {
	sc := bufio.NewScanner(r)
	// Increase buffer for very long lines (1 MiB)
	buf := make([]byte, 0, 1024*1024)
	sc.Buffer(buf, 1024*1024)

	out := make([]float64, 0, 1024)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		s := strings.TrimSpace(sc.Text())
		if s == "" {
			continue
		}
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "WARN: skipping non-numeric line %d: %q\n", lineNo, s)
			continue
		}
		out = append(out, v)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func readNumbersFromFile(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readNumbers(f)
}

// -----------------------------
// CLI glue
// -----------------------------

func usage(prog string) {
	base := filepath.Base(prog)
	fmt.Printf("Usage:\n")
	fmt.Printf("  %s <target_number> <file_with_one_number_per_line>\n", base)
	fmt.Printf("  %s <target_number>  # reads numbers from STDIN\n\n", base)
	fmt.Printf("Example:\n")
	fmt.Printf("  printf \"1\\n3\\n4\\n7\\n9\\n11\\n15\\n\" | %s 11\n", base)
}

func parseArgs() (float64, string, error) {
	if len(os.Args) < 2 {
		return 0, "", errors.New("missing target_number")
	}
	target, err := strconv.ParseFloat(os.Args[1], 64)
	if err != nil {
		return 0, "", fmt.Errorf("target_number must be numeric, got %q", os.Args[1])
	}
	file := ""
	if len(os.Args) >= 3 {
		file = os.Args[2]
	}
	return target, file, nil
}

func main() {
	target, file, err := parseArgs()
	if err != nil {
		fmt.Fprintln(os.Stderr, "ERROR:", err)
		usage(os.Args[0])
		os.Exit(2)
	}

	var nums []float64
	if file == "" {
		nums, err = readNumbers(os.Stdin)
	} else {
		nums, err = readNumbersFromFile(file)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "ERROR:", err)
		os.Exit(1)
	}
	if len(nums) == 0 {
		fmt.Fprintln(os.Stderr, "ERROR: no numeric input provided")
		os.Exit(1)
	}
	if err := ensureAscending(nums); err != nil {
		fmt.Fprintln(os.Stderr, "ERROR:", err)
		os.Exit(1)
	}

	idx, found, ins := binarySearch(nums, target)
	if found {
		fmt.Printf("FOUND %g at index %d (1-based)\n", target, idx)
	} else {
		left := "-inf"
		if ins-2 >= 0 {
			left = fmt.Sprintf("%.10g", nums[ins-2])
		}
		right := "+inf"
		if ins-1 < len(nums) {
			right = fmt.Sprintf("%.10g", nums[ins-1])
		}
		fmt.Printf("NOT FOUND %g. Insertion index %d (1-based), between %s and %s\n",
			target, ins, left, right)
	}
}

