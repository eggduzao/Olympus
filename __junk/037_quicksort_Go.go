// quicksort_inplace.go
// In-place quicksort on an array in Go.
//
// - If called with a filename, reads one number per line from that file.
// - Otherwise, reads numbers from STDIN (one per line).
// - Skips blank lines and lines beginning with '#'.
// - Prints the sorted numbers (one per line) to STDOUT.
//
// Usage (file):
//   go run quicksort_inplace.go input.txt
//
// Usage (stdin):
//   printf "5\n3\n8\n1\n2\n" | go run quicksort_inplace.go
//
// Notes:
// - Uses Hoare partition scheme and an explicit stack to avoid recursion limits.
// - Sorts []float64. Integer-looking values are printed without decimals.

package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func main() {
	var nums []float64
	var err error

	if len(os.Args) >= 2 {
		nums, err = readNumbersFromFile(os.Args[1])
		if err != nil {
			die("failed to read numbers from %q: %v", os.Args[1], err)
		}
	} else {
		nums, err = readNumbersFromReader(os.Stdin)
		if err != nil {
			die("failed to read numbers from stdin: %v", err)
		}
	}

	quickSortInPlace(nums)

	// Print one per line; show integers without trailing .000000
	w := bufio.NewWriter(os.Stdout)
	for _, x := range nums {
		if !math.IsNaN(x) && !math.IsInf(x, 0) && x == math.Trunc(x) {
			fmt.Fprintf(w, "%.0f\n", x)
		} else {
			// default fmt uses sufficient precision for float64
			fmt.Fprintf(w, "%v\n", x)
		}
	}
	w.Flush()
}

func die(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

func readNumbersFromFile(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readNumbersFromReader(f)
}

func readNumbersFromReader(r *os.Reader) ([]float64, error) {
	// Accept both *os.File and io.Reader
	var reader *bufio.Scanner
	switch v := any(*r).(type) {
	case *os.File:
		reader = bufio.NewScanner(v)
	default:
		reader = bufio.NewScanner(any(*r).(interface{ Read([]byte) (int, error) }))
	}

	// Increase the buffer in case of very long lines (though we expect short ones)
	const maxCap = 1024 * 1024 // 1MB per line
	reader.Buffer(make([]byte, 64*1024), maxCap)

	out := make([]float64, 0, 1024)
	for reader.Scan() {
		s := strings.TrimSpace(reader.Text())
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		// allow commas as thousand separators? No — keep strict; but allow simple floats.
		f64, err := strconv.ParseFloat(s, 64)
		if err != nil {
			// Include a hint where the error happened
			return nil, fmt.Errorf("cannot parse %q as number", s)
		}
		out = append(out, f64)
	}
	if err := reader.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// quickSortInPlace sorts slice A in-place using Hoare partition and an explicit stack.
func quickSortInPlace(A []float64) {
	n := len(A)
	if n <= 1 {
		return
	}

	type seg struct{ lo, hi int }
	stack := make([]seg, 0, 64)
	stack = append(stack, seg{0, n - 1})

	for len(stack) > 0 {
		// pop
		s := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		lo, hi := s.lo, s.hi

		for lo < hi {
			p := hoarePartition(A, lo, hi)

			// Tail-call elimination: push larger side, iterate on smaller side.
			leftLen := p - lo + 1
			rightLen := hi - (p + 1) + 1
			if leftLen < rightLen {
				// push larger: right
				if p+1 < hi {
					stack = append(stack, seg{p + 1, hi})
				}
				// loop on left
				hi = p
			} else {
				// push larger: left
				if lo < p {
					stack = append(stack, seg{lo, p})
				}
				// loop on right
				lo = p + 1
			}
		}
	}
}

// hoarePartition partitions A[lo:hi] using Hoare scheme, returns an index p
// such that all elements in A[lo:p] <= pivot and A[p+1:hi] >= pivot.
func hoarePartition(A []float64, lo, hi int) int {
	pivot := A[lo+(hi-lo)/2]
	i := lo - 1
	j := hi + 1
	for {
		for {
			i++
			if !(A[i] < pivot) {
				break
			}
		}
		for {
			j--
			if !(A[j] > pivot) {
				break
			}
		}
		if i >= j {
			return j
		}
		A[i], A[j] = A[j], A[i]
	}
}

// Optional helper if you want to infer input from executable name (not used above).
func programName() string { return filepath.Base(os.Args[0]) }

