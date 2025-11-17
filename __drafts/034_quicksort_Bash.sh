#!/usr/bin/env bash
# quicksort_inplace.sh
# In-place quicksort on an integer array in pure Bash.
# - Reads one integer per line from a file or stdin.
# - Sorts in ascending order in-place (Bash array).
# - Prints the sorted numbers, one per line.
#
# Usage:
#   bash quicksort_inplace.sh input.txt
#   cat input.txt | bash quicksort_inplace.sh
#
# Notes:
# - Handles duplicates and negative integers.
# - Uses Hoare partition scheme and recursion on index ranges.
# - Bash recursion depth is limited; for *huge* inputs consider tail-recursion
#   elimination or an explicit stack. For thousands of items this is fine.

# ---------- input ----------
read_input() {
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines; trim spaces
    [[ -z "${line//[[:space:]]/}" ]] && continue
    # Validate integer (optional leading sign)
    if [[ "$line" =~ ^[[:space:]]*[-+]?[0-9]+[[:space:]]*$ ]]; then
      A+=($((line)))   # normalize to integer
    else
      printf 'Warning: skipping non-integer line: %q\n' "$line" >&2
    fi
  done
}

# ---------- swap ----------
swap() {
  local i=$1 j=$2 tmp
  tmp=${A[i]}
  A[i]=${A[j]}
  A[j]=$tmp
}

# ---------- Hoare partition ----------
# Partitions A[lo..hi] around pivot A[mid], returns via echo the index "j"
# such that:
#   - all elements in A[lo..j] <= pivot
#   - all elements in A[j+1..hi] >= pivot
partition_hoare() {
  local lo=$1 hi=$2
  local mid=$(((lo+hi)/2))
  local pivot=${A[mid]}
  local i=$((lo-1))
  local j=$((hi+1))
  while :; do
    # move i right
    while :; do
      i=$((i+1))
      (( A[i] < pivot )) || break
    done
    # move j left
    while :; do
      j=$((j-1))
      (( A[j] > pivot )) || break
    done
    # stop when pointers cross
    (( i >= j )) && { echo "$j"; return; }
    swap "$i" "$j"
  done
}

# ---------- quicksort (recursive) ----------
qsort() {
  local lo=$1 hi=$2
  (( lo >= hi )) && return

  local p
  p=$(partition_hoare "$lo" "$hi")  # p is split index
  # Recurse on the smaller side first to reduce max depth (tail-call-ish)
  local left_size=$((p - lo + 1))
  local right_size=$((hi - (p+1) + 1))

  if (( left_size < right_size )); then
    qsort "$lo" "$p"
    qsort "$((p+1))" "$hi"
  else
    qsort "$((p+1))" "$hi"
    qsort "$lo" "$p"
  fi
}

# ---------- main ----------
declare -a A=()

if [[ $# -ge 1 && -n "$1" && -f "$1" ]]; then
  <"$1" read_input
else
  # read from stdin
  read_input
fi

# Nothing to do for empty input
((${#A[@]} == 0)) && exit 0

qsort 0 $((${#A[@]} - 1))

# output
for x in "${A[@]}"; do
  printf '%s\n' "$x"
done

