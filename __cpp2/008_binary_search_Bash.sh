#!/usr/bin/env bash
# binary_search.sh — Binary search over a sorted list of integers.
# Usage:
#   ./binary_search.sh TARGET [FILE]
#     TARGET : integer to find
#     FILE   : (optional) path to a newline-delimited, ASCENDING-sorted list of ints.
#              If omitted, the list is read from STDIN.
#
# Notes:
#   • Runs a classic binary search (O(log N)) implemented in awk.
#   • Returns exit code 0 on found, 1 on not found, 2 on bad input.
#   • Prints 1-based index when found; otherwise an insertion index hint.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 TARGET [FILE]" >&2
  exit 2
fi

target="$1"
input="${2:-/dev/stdin}"

# Validate that TARGET is an integer
if ! [[ "$target" =~ ^[+-]?[0-9]+$ ]]; then
  echo "ERROR: TARGET must be an integer, got: '$target'" >&2
  exit 2
fi

awk -v target="$target" '
  # Accept lines that are just an integer (optionally with whitespace)
  /^[[:space:]]*[+-]?[0-9]+[[:space:]]*$/ {
    n++; a[n] = $1
    next
  }

  # Ignore completely empty lines; complain about junk
  /^[[:space:]]*$/ { next }

  {
    printf("WARNING: skipping non-integer line %d: %s\n", NR, $0) > "/dev/stderr"
  }

  END {
    if (n == 0) {
      print "ERROR: no integers to search (after filtering)." > "/dev/stderr"
      exit 2
    }

    # Optional monotonicity guard: detect obvious unsorted input
    for (i = 2; i <= n; i++) {
      if (a[i] < a[i-1]) {
        print "ERROR: input is not in ascending order (required for binary search)." > "/dev/stderr"
        exit 2
      }
    }

    # Binary search for FIRST occurrence (if duplicates exist)
    lo = 1
    hi = n
    found = -1

    while (lo <= hi) {
      mid = int((lo + hi) / 2)
      val = a[mid]

      if (val == target) {
        found = mid
        hi = mid - 1   # keep searching left for first occurrence
      } else if (val < target) {
        lo = mid + 1
      } else {
        hi = mid - 1
      }
    }

    if (found != -1) {
      printf("FOUND %d at index %d (1-based)\n", target, found)
      exit 0
    } else {
      # lo = insertion point (1-based) that preserves order
      left  = (lo > 1) ? a[lo-1] : "-inf"
      right = (lo <= n) ? a[lo]   : "+inf"
      printf("NOT FOUND %d. Insertion index %d (1-based), between %s and %s\n",
             target, lo, left, right)
      exit 1
    }
  }
' "$input"

