#!/usr/bin/env Rscript
# In-place quicksort on an array (R version).
# - Reads numbers (ints or floats) one-per-line from a file path (arg1) or stdin.
# - Ignores blank lines and lines starting with '#'.
# - Sorts using iterative quicksort with Hoare partition + median-of-three pivot.
# - Small ranges use insertion sort (faster in practice).
# - Prints sorted values one per line (integers printed without ".0").
#
# Usage:
#   Rscript quicksort_inplace.R numbers.txt
#   cat numbers.txt | Rscript quicksort_inplace.R

read_numbers <- function(con) {
  lines <- readLines(con, warn = FALSE)
  lines <- trimws(lines)
  lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
  if (length(lines) == 0L) return(numeric(0))

  # Parse as numeric; fail on non-finite or non-numeric
  suppressWarnings(nums <- as.numeric(lines))
  bad <- which(!is.finite(nums))
  if (length(bad)) {
    stop(sprintf("Invalid numeric line at %d: %s",
                 bad[1], lines[bad[1]]), call. = FALSE)
  }
  nums
}

# Pretty print: integers without trailing .0
print_numbers <- function(x) {
  is_int <- abs(x - round(x)) < .Machine$double.eps * 4
  out <- ifelse(is_int, as.character(round(x)), format(x, scientific = FALSE, trim = TRUE))
  cat(paste0(out, collapse = "\n"), "\n", sep = "")
}

# --------- Sorting primitives (1-based indexing) ---------

median_of_three <- function(a, i, j, k) {
  ai <- a[i]; aj <- a[j]; ak <- a[k]
  if (ai < aj) {
    if (aj < ak) j else if (ai < ak) i else k
  } else {
    if (ai < ak) i else if (aj < ak) j else k
  }
}

insertion_sort <- function(a, lo, hi) {
  if (lo >= hi) return(a)
  for (i in (lo + 1L):hi) {
    key <- a[i]
    j <- i - 1L
    while (j >= lo && a[j] > key) {
      a[j + 1L] <- a[j]
      j <- j - 1L
    }
    a[j + 1L] <- key
  }
  a
}

# Hoare partition: returns p so that [lo..p] <= pivot and [p+1..hi] >= pivot
hoare_partition <- function(a, lo, hi) {
  mid  <- lo + ((hi - lo) %/% 2L)
  pidx <- median_of_three(a, lo, mid, hi)
  pivot <- a[pidx]
  i <- lo - 1L
  j <- hi + 1L
  repeat {
    repeat { i <- i + 1L; if (a[i] >= pivot) break }
    repeat { j <- j - 1L; if (a[j] <= pivot) break }
    if (i >= j) return(list(a = a, p = j))
    tmp <- a[i]; a[i] <- a[j]; a[j] <- tmp
  }
}

# Iterative quicksort with small-range cutoff
quicksort_inplace <- function(a) {
  n <- length(a)
  if (n < 2L) return(a)

  SMALL <- 32L
  stack_lo <- integer(0); stack_hi <- integer(0)
  push <- function(lo, hi) { 
    stack_lo <<- c(stack_lo, lo); stack_hi <<- c(stack_hi, hi)
  }
  pop <- function() {
    lo <- stack_lo[length(stack_lo)]
    hi <- stack_hi[length(stack_hi)]
    stack_lo <<- stack_lo[-length(stack_lo)]
    stack_hi <<- stack_hi[-length(stack_hi)]
    c(lo, hi)
  }

  push(1L, n)

  while (length(stack_lo)) {
    rng <- pop(); lo <- rng[1]; hi <- rng[2]
    if (hi - lo + 1L <= SMALL) {
      a <- insertion_sort(a, lo, hi)
      next
    }
    part <- hoare_partition(a, lo, hi)
    a <- part$a; p <- part$p
    left_sz  <- p - lo + 1L
    right_sz <- hi - (p + 1L) + 1L

    # Process smaller side first (reduces stack depth)
    if (left_sz < right_sz) {
      if (p + 1L < hi) push(p + 1L, hi)
      if (lo < p)      push(lo, p)
    } else {
      if (lo < p)      push(lo, p)
      if (p + 1L < hi) push(p + 1L, hi)
    }
  }
  a
}

# --------- Main ---------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1L) {
  con <- file(args[1], open = "r")
  on.exit(close(con), add = TRUE)
  nums <- read_numbers(con)
} else {
  nums <- read_numbers("stdin")
}

sorted <- quicksort_inplace(nums)
print_numbers(sorted)

