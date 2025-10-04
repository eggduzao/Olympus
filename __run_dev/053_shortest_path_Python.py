#!/usr/bin/env python3
"""
BFS shortest path on an unweighted graph (single-file script).

Input format (flexible):
- Each non-empty, non-comment line lists a node followed by its neighbors.
  Delimiters: tabs / spaces / commas are all accepted.
  Example (TSV-like):
      A   B   F
      B   A   C
- Query lines start with '#', followed by source and target:
      #  A  E
  (You can include multiple query lines; each will be answered.)

CLI:
  python bfs_shortest_path.py graph.txt
  python bfs_shortest_path.py --from A --to E graph.txt
  cat graph.txt | python bfs_shortest_path.py -           # read from stdin
  python bfs_shortest_path.py --directed graph.txt        # treat edges as directed

Notes:
- Defaults to UNDIRECTED unless --directed is set.
- If no query lines are present, you must provide --from and --to.
- Handles large graphs (stream parsing, O(V+E) memory/time).
"""

from __future__ import annotations
import sys
import argparse
import re
from collections import deque, defaultdict
from typing import Dict, Iterable, List, Tuple, Deque, Optional, Set

Adj = Dict[str, Set[str]]

def parse_graph_and_queries(lines: Iterable[str], directed: bool = False) -> Tuple[Adj, List[Tuple[str, str]]]:
    graph: Adj = defaultdict(set)
    queries: List[Tuple[str, str]] = []
    splitter = re.compile(r"[,\t ]+")  # allow commas, tabs, spaces

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Query line(s): "# src dst"
        if line.startswith("#"):
            tokens = splitter.split(line[1:].strip())
            if len(tokens) >= 2:
                src, dst = tokens[0], tokens[1]
                queries.append((src, dst))
            # allow comment-only lines; ignore if fewer than 2 tokens
            continue

        parts = splitter.split(line)
        if not parts or parts[0] == "":
            continue

        node = parts[0]
        neighbors = parts[1:]

        # Ensure node exists even if it has no neighbors listed
        _ = graph[node]

        for nb in neighbors:
            if nb == "":
                continue
            graph[node].add(nb)
            if not directed:
                graph[nb].add(node)  # ensure reverse edge present and nb exists

    return graph, queries

def bfs_shortest_path(graph: Adj, src: str, dst: str) -> Optional[List[str]]:
    """
    Standard BFS on an unweighted graph to retrieve a shortest path by hop-count.
    Returns the node list from src to dst inclusive, or None if unreachable.
    """
    if src not in graph or dst not in graph:
        return None
    if src == dst:
        return [src]

    q: Deque[str] = deque([src])
    visited: Set[str] = {src}
    parent: Dict[str, str] = {}

    while q:
        u = q.popleft()
        for v in graph[u]:
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            if v == dst:
                # reconstruct
                path = [dst]
                while path[-1] != src:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            q.append(v)

    return None

def format_path(path: List[str]) -> str:
    return " -> ".join(path)

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="BFS shortest path on an unweighted graph.")
    p.add_argument("input", help="Path to input file, or '-' for stdin.")
    p.add_argument("--from", dest="src", help="Source node (used if no '# src dst' query lines).")
    p.add_argument("--to", dest="dst", help="Target node (used if no '# src dst' query lines).")
    p.add_argument("--directed", action="store_true", help="Treat listed edges as directed (default: undirected).")
    args = p.parse_args(argv)

    # Read input
    if args.input == "-":
        lines = sys.stdin
    else:
        try:
            lines = open(args.input, "r", encoding="utf-8")
        except OSError as e:
            print(f"ERROR: cannot open input file '{args.input}': {e}", file=sys.stderr)
            return 2

    try:
        graph, queries = parse_graph_and_queries(lines, directed=args.directed)
    finally:
        if args.input != "-":
            try:
                lines.close()  # type: ignore
            except Exception:
                pass

    # If no inline queries, use CLI src/dst
    if not queries:
        if not args.src or not args.dst:
            print("ERROR: no queries found in input and --from/--to not provided.", file=sys.stderr)
            return 3
        queries = [(args.src, args.dst)]

    # Answer queries
    # For reproducibility, we sort adjacency sets during BFS iteration implicitly by constructing lists,
    # but that’s optional; BFS correctness does not rely on neighbor order.
    # (If you want deterministic tie-breaking, you could convert sets to sorted lists in bfs.)
    for (src, dst) in queries:
        path = bfs_shortest_path(graph, src, dst)
        if path is None:
            print(f"NO PATH: {src} -> {dst}")
        else:
            hops = len(path) - 1
            print(f"PATH ({hops} edges): {format_path(path)}")

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
