// bfs_unweighted.go
// Breadth-first search (BFS) shortest path on an unweighted, undirected graph.
//
// INPUT FORMAT (tab- or space-separated; one edge per line; queries start with '#'):
//   A   B   F
//   B   A   C
//   C   B   D
//   D   C   E
//   E   D   F
//   F   A   E
//   #   A   E
//
// USAGE
//   go run bfs_unweighted.go graph.tsv
//   # or pipe:
//   cat graph.tsv | go run bfs_unweighted.go
//
// OUTPUT
//   One line per query, either "SRC -> ... -> DST" or an explanatory message.

package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

type pair struct{ src, dst string }

func main() {
	var in *os.File
	if len(os.Args) >= 2 {
		f, err := os.Open(os.Args[1])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to open %s: %v\n", os.Args[1], err)
			os.Exit(1)
		}
		defer f.Close()
		in = f
	} else {
		in = os.Stdin
	}

	adj, queries := parseInput(in)
	if len(queries) == 0 {
		fmt.Println(`No queries found (expect lines like: "#  SRC  DST"). Nothing to do.`)
		return
	}

	for _, q := range queries {
		if q.src == q.dst {
			fmt.Println(q.src)
			continue
		}
		if _, ok := adj[q.src]; !ok {
			fmt.Printf("No path found from %s to %s (source not in graph)\n", q.src, q.dst)
			continue
		}
		if _, ok := adj[q.dst]; !ok {
			fmt.Printf("No path found from %s to %s (destination not in graph)\n", q.src, q.dst)
			continue
		}
		path := bfsPath(adj, q.src, q.dst)
		if len(path) == 0 {
			fmt.Printf("No path found from %s to %s\n", q.src, q.dst)
		} else {
			fmt.Println(strings.Join(path, " -> "))
		}
	}
}

//--------------------------- Parsing & Utilities ---------------------------//

// parseInput reads whitespace-separated tokens per line. Lines that start with
// '#' are queries "# SRC DST". Other lines encode edges "U V1 [V2 ...]"
// and are treated as undirected edges U—V1, U—V2, ...
func parseInput(in *os.File) (map[string][]string, []pair) {
	adj := make(map[string][]string)
	var queries []pair

	sc := bufio.NewScanner(in)
	// Increase buffer for very long lines (e.g., many neighbors)
	const maxCapacity = 4 << 20 // 4 MiB
	buf := make([]byte, 0, 64<<10)
	sc.Buffer(buf, maxCapacity)

	for lineNum := 1; sc.Scan(); lineNum++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		// Normalize internal whitespace
		toks := strings.Fields(line)
		if len(toks) == 0 {
			continue
		}
		if strings.HasPrefix(toks[0], "#") {
			if len(toks) < 3 {
				// Graceful skip of malformed query
				continue
			}
			queries = append(queries, pair{src: toks[1], dst: toks[2]})
			continue
		}
		if len(toks) < 2 {
			// Malformed edge line; skip
			continue
		}
		u := toks[0]
		for _, v := range toks[1:] {
			addUndirectedEdge(adj, u, v)
		}
	}
	// Note: ignoring sc.Err() to keep output clean for pipelines; in production, handle it.

	return adj, queries
}

func addUndirectedEdge(adj map[string][]string, u, v string) {
	addDirectedUnique(adj, u, v)
	addDirectedUnique(adj, v, u)
}

func addDirectedUnique(adj map[string][]string, u, v string) {
	nb := adj[u]
	// Deduplicate neighbor entries without allocating a set
	for _, x := range nb {
		if x == v {
			return
		}
	}
	adj[u] = append(nb, v)
}

//--------------------------- BFS Shortest Path -----------------------------//

// bfsPath returns the shortest path from src to dst as a slice of node labels,
// or an empty slice if no path exists. Graph is unweighted & undirected.
func bfsPath(adj map[string][]string, src, dst string) []string {
	if src == dst {
		return []string{src}
	}

	visited := make(map[string]bool, len(adj))
	parent := make(map[string]string, len(adj))

	// Simple queue using a slice with head index to avoid O(n) pops.
	q := make([]string, 0, 1024)
	head := 0
	q = append(q, src)
	visited[src] = true

	for head < len(q) {
		u := q[head]
		head++

		for _, v := range adj[u] {
			if !visited[v] {
				visited[v] = true
				parent[v] = u
				if v == dst {
					return reconstruct(parent, src, dst)
				}
				q = append(q, v)
			}
		}
	}
	return nil
}

func reconstruct(parent map[string]string, src, dst string) []string {
	path := []string{dst}
	cur := dst
	for cur != src {
		p, ok := parent[cur]
		if !ok {
			return nil // safety guard
		}
		cur = p
		path = append(path, cur)
	}
	// reverse in-place
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}

