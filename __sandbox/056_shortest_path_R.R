#!/usr/bin/env Rscript
# BFS shortest path on an unweighted graph — single-file R script
# Input lines:
#   node <delim> neighbor1 <delim> neighbor2 ...
#   Delims: spaces, tabs, or commas. Blank lines ignored.
#   Query lines start with '# src dst' (same delimiters). You may include many.
#   If no query lines are present, provide --from SRC --to DST on the CLI.
#
# Examples:
#   Rscript bfs_shortest_path.R graph.txt
#   Rscript bfs_shortest_path.R --from A --to E graph.txt
#   cat graph.txt | Rscript bfs_shortest_path.R -
#
# Options:
#   --directed    treat edges as directed (default: undirected)
#   Input path '-' means stdin.

delims <- "[, \t]+"

die <- function(msg, status = 2L) {
  message("ERROR: ", msg)
  quit(status = status, save = "no")
}

parse_args <- function(argv) {
  a <- list(directed = FALSE, input = "-", src = NULL, dst = NULL)
  i <- 1L
  while (i <= length(argv)) {
    t <- argv[[i]]
    if (t == "--directed") {
      a$directed <- TRUE
    } else if (t == "--from") {
      if (i + 1L > length(argv)) die("Missing value after --from")
      i <- i + 1L; a$src <- argv[[i]]
    } else if (t == "--to") {
      if (i + 1L > length(argv)) die("Missing value after --to")
      i <- i + 1L; a$dst <- argv[[i]]
    } else {
      if (a$input == "-") a$input <- t else die(paste("Unexpected argument:", t))
    }
    i <- i + 1L
  }
  a
}

read_lines <- function(path) {
  if (identical(path, "-")) {
    readLines(file("stdin"), warn = FALSE)
  } else {
    if (!file.exists(path)) die(paste("Input not found:", path))
    readLines(path, warn = FALSE)
  }
}

ensure_node <- function(g, n) {
  if (is.null(g[[n]])) g[[n]] <- character(0)
  g
}

add_edge <- function(g, u, v, directed = FALSE) {
  g <- ensure_node(g, u); g <- ensure_node(g, v)
  g[[u]] <- unique(c(g[[u]], v))
  if (!directed) g[[v]] <- unique(c(g[[v]], u))
  g
}

parse_graph_queries <- function(lines, directed = FALSE) {
  g <- new.env(parent = emptyenv()); g[] <- list()
  queries <- list()
  for (raw in lines) {
    if (is.na(raw)) next
    line <- trimws(raw)
    if (nchar(line) == 0L) next
    if (substr(line, 1L, 1L) == "#") {
      rest <- trimws(sub("^#", "", line))
      if (nchar(rest) > 0L) {
        toks <- unlist(strsplit(rest, delims, perl = TRUE))
        if (length(toks) >= 2L) queries[[length(queries) + 1L]] <- c(toks[1L], toks[2L])
      }
      next
    }
    toks <- unlist(strsplit(line, delims, perl = TRUE))
    if (length(toks) == 0L) next
    u <- toks[1L]
    g[[u]] <- if (is.null(g[[u]])) character(0) else g[[u]]
    if (length(toks) >= 2L) {
      for (v in toks[-1L]) {
        if (nzchar(v)) g <- add_edge(g, u, v, directed = directed)
      }
    } else {
      # isolated node
      g <- ensure_node(g, u)
    }
  }
  # materialize env -> named list
  graph <- as.list(g)
  list(graph = graph, queries = queries)
}

bfs_shortest_path <- function(graph, src, dst) {
  if (is.null(graph[[src]]) || is.null(graph[[dst]])) return(NULL)
  if (identical(src, dst)) return(src)

  # build integer ids for compact queue ops
  nodes <- names(graph)
  id <- seq_along(nodes); names(id) <- nodes
  N <- length(nodes)
  parent <- rep.int(NA_integer_, N)
  seen <- rep.int(FALSE, N)
  q <- integer(N); head <- 1L; tail <- 0L

  s <- id[[src]]; t <- id[[dst]]
  seen[s] <- TRUE; tail <- tail + 1L; q[tail] <- s

  while (head <= tail) {
    u <- q[head]; head <- head + 1L
    unode <- nodes[[u]]
    neighs <- graph[[unode]]
    if (length(neighs)) {
      for (vname in neighs) {
        v <- id[[vname]]
        if (isTRUE(seen[v])) next
        seen[v] <- TRUE
        parent[v] <- u
        if (v == t) {
          # reconstruct
          path_ids <- integer(0)
          cur <- v
          while (!is.na(cur)) {
            path_ids <- c(cur, path_ids)
            if (cur == s) break
            cur <- parent[cur]
          }
          if (path_ids[1L] != s) return(NULL)
          return(nodes[path_ids])
        }
        tail <- tail + 1L; q[tail] <- v
      }
    }
  }
  NULL
}

format_path <- function(path) paste(path, collapse = " -> ")

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  a <- parse_args(args)
  lines <- read_lines(a$input)
  parsed <- parse_graph_queries(lines, directed = a$directed)

  queries <- parsed$queries
  if (length(queries) == 0L) {
    if (is.null(a$src) || is.null(a$dst)) {
      die("No queries in input and --from/--to not provided.")
    }
    queries <- list(c(a$src, a$dst))
  }

  for (q in queries) {
    src <- q[[1L]]; dst <- q[[2L]]
    path <- bfs_shortest_path(parsed$graph, src, dst)
    if (is.null(path)) {
      cat(sprintf("NO PATH: %s -> %s\n", src, dst))
    } else {
      hops <- length(path) - 1L
      cat(sprintf("PATH (%d edges): %s\n", hops, format_path(path)))
    }
  }
}

if (identical(environment(), globalenv())) main()

