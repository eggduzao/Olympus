<?php
/**
 * bfs_shortest_path.php
 *
 * Breadth-First Search (BFS) shortest path on an unweighted, UNDIRECTED graph.
 *
 * INPUT FORMAT (whitespace/tab separated):
 *   Edge lines : "U  V1 [V2 ...]"  => add undirected edges U—V1, U—V2, ...
 *   Query line : "#  SRC DST"      => request one shortest path from SRC to DST
 *
 * Example:
 *   A   B   F
 *   B   A   C
 *   C   B   D
 *   D   C   E
 *   E   D   F
 *   F   A   E
 *   #   A   E
 *
 * USAGE
 *   # stdin:
 *   php bfs_shortest_path.php < graph.tsv
 *
 *   # or with a file path:
 *   php bfs_shortest_path.php path/to/graph.tsv
 *
 * OUTPUT
 *   For each query line: either a path like "A -> B -> C" or a "No path found..." message.
 */

declare(strict_types=1);

/* ------------------------------------------------------------- */
/* Entry point                                                   */
/* ------------------------------------------------------------- */
if (php_sapi_name() === 'cli' && realpath($argv[0]) === __FILE__) {
    [$graph, $queries] = parse_input($argv);

    if (empty($queries)) {
        fwrite(STDERR, "No queries found (expected lines like: \"# SRC DST\").\n");
        exit(0);
    }

    foreach ($queries as [$src, $dst]) {
        $path = bfs_shortest_path($graph, $src, $dst);
        if ($path === null) {
            echo "No path found from {$src} to {$dst}\n";
        } else {
            echo implode(' -> ', $path) . PHP_EOL;
        }
    }
}

/* ------------------------------------------------------------- */
/* Parsing                                                       */
/* ------------------------------------------------------------- */

/**
 * @param array $argv
 * @return array{0: array<string, array<int, string>>, 1: array<int, array{0:string,1:string}>}
 */
function parse_input(array $argv): array
{
    // Read all lines from either file argument or STDIN
    if (isset($argv[1])) {
        $contents = file_get_contents($argv[1]);
        if ($contents === false) {
            fwrite(STDERR, "Failed to read file: {$argv[1]}\n");
            exit(1);
        }
    } else {
        $contents = stream_get_contents(STDIN);
        if ($contents === false) {
            fwrite(STDERR, "Failed to read STDIN\n");
            exit(1);
        }
    }

    $graphSets = []; // node => set (assoc array) of neighbors for dedupe
    $queries   = []; // list of [src, dst]

    $lines = preg_split('/\R/u', $contents) ?: [];
    foreach ($lines as $lnRaw) {
        $ln = trim($lnRaw);
        if ($ln === '') {
            continue;
        }
        // Split on any run of whitespace (spaces or tabs)
        $parts = preg_split('/\s+/u', $ln);
        if (!$parts || count($parts) === 0) {
            continue;
        }
        if ($parts[0] === '#') {
            if (count($parts) >= 3) {
                $queries[] = [$parts[1], $parts[2]];
            }
            continue;
        }

        // Edge line: U V1 V2 ...
        $u = $parts[0];
        ensure_node($graphSets, $u);
        for ($i = 1; $i < count($parts); $i++) {
            $v = $parts[$i];
            ensure_node($graphSets, $v);
            // undirected add with dedupe
            $graphSets[$u][$v] = true;
            $graphSets[$v][$u] = true;
        }
    }

    // Convert neighbor "sets" to sorted arrays for stable behavior
    $graph = [];
    foreach ($graphSets as $node => $set) {
        $neighbors = array_keys($set);
        sort($neighbors, SORT_STRING);
        $graph[$node] = $neighbors;
    }

    return [$graph, $queries];
}

/**
 * Ensure node key exists with empty "set".
 * @param array<string, array<string,bool>> &$g
 * @param string $u
 * @return void
 */
function ensure_node(array &$g, string $u): void
{
    if (!array_key_exists($u, $g)) {
        $g[$u] = [];
    }
}

/* ------------------------------------------------------------- */
/* BFS shortest path                                             */
/* ------------------------------------------------------------- */

/**
 * BFS shortest path from $src to $dst.
 *
 * @param array<string, array<int, string>> $graph adjacency list
 * @param string $src
 * @param string $dst
 * @return array<int, string>|null  [src,...,dst] or null if unreachable
 */
function bfs_shortest_path(array $graph, string $src, string $dst): ?array
{
    if ($src === $dst) {
        return [$src];
    }
    if (!array_key_exists($src, $graph) || !array_key_exists($dst, $graph)) {
        return null;
    }

    // SplQueue is an efficient linked-list queue in PHP
    $q = new SplQueue();
    $q->enqueue($src);

    $visited = [$src => true];
    /** @var array<string, string> $parent */
    $parent  = [];

    while (!$q->isEmpty()) {
        /** @var string $u */
        $u = $q->dequeue();
        $neighbors = $graph[$u] ?? [];
        foreach ($neighbors as $v) {
            if (isset($visited[$v])) {
                continue;
            }
            $visited[$v] = true;
            $parent[$v]  = $u;

            if ($v === $dst) {
                return reconstruct_path($parent, $src, $dst);
            }
            $q->enqueue($v);
        }
    }
    return null;
}

/**
 * Reconstruct path from $src to $dst using $parent map.
 *
 * @param array<string, string> $parent
 * @param string $src
 * @param string $dst
 * @return array<int, string>
 */
function reconstruct_path(array $parent, string $src, string $dst): array
{
    $path = [];
    for ($cur = $dst; $cur !== $src; $cur = $parent[$cur] ?? null) {
        if ($cur === null) {
            // Safety: should not happen if called correctly
            return [];
        }
        $path[] = $cur;
    }
    $path[] = $src;
    return array_reverse($path);
}

