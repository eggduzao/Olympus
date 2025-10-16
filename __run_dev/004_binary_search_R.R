# binary_search.R — Binary search in a sorted array (single-file R script)
#
# Usage:
#   Rscript binary_search.R input.txt 1 9 15 2
#
#   - input.txt: text file with one number per line (unsorted OK; we sort)
#   - following args are targets to query (if none given, uses a demo set)
#
# Notes:
#   • Time: O(log n) per query after O(n log n) sort.
#   • Space: O(n) to hold the vector. For multi-GB inputs, consider an
#     on-disk index (SQLite, DuckDB) instead of reading all values into RAM.

suppressWarnings({
  options(warn = 1, scipen = 999)
})

# ---- Core binary search (ascending, 0-based logic internally for clarity) ----
bin_search <- function(x, target) {
  # x must be sorted ascending (numeric). Returns 1-based index if found, else -1.
  lo <- 1L
  hi <- length(x)
  while (lo <= hi) {
    mid <- lo + as.integer((hi - lo) %/% 2L)
    v <- x[mid]
    if (is.na(v)) return(-1L)  # shouldn't happen in cleaned vector
    if (v == target) return(mid)
    if (v < target) {
      lo <- mid + 1L
    } else {
      hi <- mid - 1L
    }
  }
  -1L
}

# ---- Load numbers from file, skip malformed lines, return sorted numeric ----
load_sorted_vector <- function(path) {
  if (!file.exists(path)) stop("Input file not found: ", path)
  # readLines is memory-bound; for very large files, consider chunked reading
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  lines <- trimws(lines)
  lines <- lines[nzchar(lines)]
  vals  <- suppressWarnings(as.numeric(lines))
  bad   <- which(is.na(vals))
  if (length(bad)) {
    for (i in bad) message(sprintf("[skip] line %d not numeric: %s", i, lines[i]))
    vals <- vals[!is.na(vals)]
  }
  vals <- sort(vals)  # ascending
  vals
}

# ---- Reporting helper ----
report_result <- function(target, idx) {
  if (idx > 0L) {
    cat(sprintf("Target %g: found at index %d\n", target, idx))
  } else {
    cat(sprintf("Target %g: not found\n", target))
  }
}

# ---- "Main" entrypoint (R's equivalent of if __name__ == '__main__') ----
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  cat("Usage: Rscript binary_search.R <input.txt> [targets ...]\n",
      "Example: Rscript binary_search.R input.txt 1 9 15 2\n", sep = "")
  quit(status = 2)
}

input_path <- args[[1]]
vec <- tryCatch(load_sorted_vector(input_path),
                error = function(e) { message("Failed to read input: ", e$message); quit(status = 1) })
cat(sprintf("Loaded %d numbers from %s\n", length(vec), input_path))

targets <- if (length(args) > 1L) {
  tvec <- suppressWarnings(as.numeric(args[-1L]))
  if (any(is.na(tvec))) {
    bad <- which(is.na(tvec))
    for (i in bad) message("[skip target] not numeric: ", args[-1L][i])
    tvec <- tvec[!is.na(tvec)]
  }
  if (!length(tvec)) c(1, 9, 15, 2) else tvec
} else {
  c(1, 9, 15, 2)  # demo defaults
}

for (t in targets) {
  idx <- bin_search(vec, t)
  report_result(t, idx)
}
