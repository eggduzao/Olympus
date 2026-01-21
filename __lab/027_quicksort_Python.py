#!/usr/bin/env python3
"""
In-place quicksort on an array (robust, iterative, median-of-three pivot, Hoare partition).

Usage
-----
# stdin (one integer per line)
$ python quicksort_inplace.py < numbers.txt

# or pass a file path
$ python quicksort_inplace.py numbers.txt

Input format
------------
- One number per line (integers or floats).
- Blank lines are ignored.
- Lines starting with '#' are treated as comments.

Output
------
- Sorted numbers, one per line.

Notes
-----
- Uses an *iterative* quicksort (manual stack) to avoid Python recursion limits.
- Hoare partition scheme + median-of-three pivot selection for fewer swaps and better behavior
  on partially-sorted data.
- Switches to insertion sort for tiny partitions to reduce overhead.
- Sorts the list *in place*.
"""

from __future__ import annotations
import sys
from typing import Iterable, List, Sequence, Tuple, TextIO, Union

Number = Union[int, float]


# --------------------------
# Parsing & I/O
# --------------------------
def _parse_numbers(stream: TextIO) -> List[Number]:
    out: List[Number] = []
    for raw in stream:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # Accept ints and floats
        try:
            n: Number
            if any(ch in s for ch in ".eE"):
                n = float(s)
            else:
                n = int(s)
        except ValueError as e:
            raise ValueError(f"Invalid numeric line {raw!r}") from e
        out.append(n)
    return out


def _print_numbers(arr: Sequence[Number]) -> None:
    w = sys.stdout.write
    for x in arr:
        w(f"{x}\n")


# --------------------------
# Sorting primitives
# --------------------------
def _median_of_three(a: List[Number], i: int, j: int, k: int) -> int:
    """Return index of median(a[i], a[j], a[k]) without extra allocations."""
    ai, aj, ak = a[i], a[j], a[k]
    # Compare in a branch-minimizing way
    if ai < aj:
        if aj < ak:
            return j  # ai < aj < ak
        return i if ai < ak else k  # ai < ak <= aj  OR  ak <= ai < aj
    else:
        if ai < ak:
            return i  # aj <= ai < ak
        return j if aj < ak else k  # aj < ak <= ai  OR  ak <= aj <= ai


def _insertion_sort(a: List[Number], lo: int, hi: int) -> None:
    """Simple insertion sort on a[lo:hi+1] (inclusive bounds)."""
    for i in range(lo + 1, hi + 1):
        key = a[i]
        j = i - 1
        while j >= lo and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


def _hoare_partition(a: List[Number], lo: int, hi: int) -> int:
    """
    Hoare partition on a[lo:hi] inclusive bounds.
    Returns the final index j such that all elements <= pivot are <= j,
    and all elements >= pivot are >= j+1.
    """
    # Median-of-three pivot selection to mitigate worst-case
    mid = lo + (hi - lo) // 2
    pidx = _median_of_three(a, lo, mid, hi)
    pivot = a[pidx]

    i, j = lo - 1, hi + 1
    while True:
        # Move i right
        i += 1
        while a[i] < pivot:
            i += 1
        # Move j left
        j -= 1
        while a[j] > pivot:
            j -= 1
        if i >= j:
            return j
        # Swap out-of-place pair
        a[i], a[j] = a[j], a[i]


def quicksort_inplace(a: List[Number]) -> None:
    """
    In-place quicksort with iterative stack, Hoare partition, median-of-three pivot,
    and insertion-sort cutoff for tiny partitions.
    """
    n = len(a)
    if n < 2:
        return

    # Tuning knobs
    SMALL = 16  # insertion sort cutoff

    # Manual stack of (lo, hi) inclusive ranges
    stack: List[Tuple[int, int]] = [(0, n - 1)]

    while stack:
        lo, hi = stack.pop()
        # For tiny partitions, defer to insertion sort
        if hi - lo + 1 <= SMALL:
            _insertion_sort(a, lo, hi)
            continue

        p = _hoare_partition(a, lo, hi)

        # Tail recursion elimination heuristic:
        # Process the smaller side first to keep stack shallow.
        left_size = p - lo + 1
        right_size = hi - (p + 1) + 1

        if left_size < right_size:
            if lo < p:
                stack.append((p + 1, hi))  # push larger side
                stack.append((lo, p))      # handle smaller side next
            else:
                stack.append((p + 1, hi))
        else:
            if p + 1 < hi:
                stack.append((lo, p))      # push larger side
                stack.append((p + 1, hi))  # handle smaller side next
            else:
                stack.append((lo, p))


# --------------------------
# Main
# --------------------------
def main(argv: List[str]) -> int:
    # Read numbers from a file or stdin
    if len(argv) > 1:
        with open(argv[1], "r", encoding="utf-8") as fh:
            arr = _parse_numbers(fh)
    else:
        arr = _parse_numbers(sys.stdin)

    quicksort_inplace(arr)
    _print_numbers(arr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

