/**
 * BFS shortest path on an unweighted graph (Node.js, single-file script).
 *
 * Input format (flexible):
 * - Each non-empty, non-comment line lists a node followed by its neighbors.
 *   Delimiters: tabs / spaces / commas are all accepted.
 *   Example (TSV-like):
 *       A   B   F
 *       B   A   C
 * - Query lines start with '#', followed by source and target:
 *       #  A  E
 *   (You can include multiple query lines; each will be answered.)
 *
 * CLI:
 *   node bfs_shortest_path.js graph.txt
 *   node bfs_shortest_path.js --from A --to E graph.txt
 *   cat graph.txt | node bfs_shortest_path.js -
 *   node bfs_shortest_path.js --directed graph.txt
 *
 * Notes:
 * - Defaults to UNDIRECTED unless --directed is set.
 * - If no query lines are present, you must provide --from and --to.
 */

'use strict';

const fs = require('fs');

function parseArgs(argv) {
  // Very small arg parser: supports --from, --to, --directed, and an input path
  const args = { directed: false, input: null, src: null, dst: null };
  const it = argv[Symbol.iterator]();
  for (let token of it) {
    if (token === '--directed') {
      args.directed = true;
    } else if (token === '--from') {
      const v = it.next().value;
      if (!v) throw new Error('Missing value after --from');
      args.src = v;
    } else if (token === '--to') {
      const v = it.next().value;
      if (!v) throw new Error('Missing value after --to');
      args.dst = v;
    } else if (!args.input) {
      args.input = token;
    } else {
      // extra positional arguments not expected
      throw new Error(`Unexpected argument: ${token}`);
    }
  }
  if (!args.input) args.input = '-'; // default to stdin if omitted
  return args;
}

function splitTokens(line) {
  // Allow commas, tabs, or spaces as delimiters; collapse multiples
  return line.trim().split(/[,\t ]+/).filter(Boolean);
}

/**
 * Build graph (as Map<string, Set<string>>) and collect queries.
 */
function parseGraphAndQueries(lines, { directed = false } = {}) {
  const graph = new Map(); // node -> Set(neighbors)
  const queries = [];

  function ensureNode(n) {
    if (!graph.has(n)) graph.set(n, new Set());
  }

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;

    if (line.startsWith('#')) {
      const toks = splitTokens(line.slice(1));
      if (toks.length >= 2) {
        queries.push([toks[0], toks[1]]);
      }
      continue;
    }

    const parts = splitTokens(line);
    if (parts.length === 0) continue;

    const node = parts[0];
    const neighbors = parts.slice(1);
    ensureNode(node);
    for (const nb of neighbors) {
      if (!nb) continue;
      ensureNode(nb);
      graph.get(node).add(nb);
      if (!directed) graph.get(nb).add(node);
    }
  }
  return { graph, queries };
}

/**
 * Standard BFS to return a shortest (fewest edges) path between src and dst.
 * Returns an array of nodes (src..dst) or null if no path.
 */
function bfsShortestPath(graph, src, dst) {
  if (!graph.has(src) || !graph.has(dst)) return null;
  if (src === dst) return [src];

  const q = [src];
  let qHead = 0; // efficient queue using index
  const visited = new Set([src]);
  const parent = new Map();

  while (qHead < q.length) {
    const u = q[qHead++];
    for (const v of graph.get(u)) {
      if (visited.has(v)) continue;
      visited.add(v);
      parent.set(v, u);
      if (v === dst) {
        // reconstruct path
        const path = [dst];
        while (path[path.length - 1] !== src) {
          path.push(parent.get(path[path.length - 1]));
        }
        path.reverse();
        return path;
      }
      q.push(v);
    }
  }
  return null;
}

function formatPath(path) {
  return path.join(' -> ');
}

function readAllLinesFromStdin() {
  return new Promise((resolve, reject) => {
    let buf = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => (buf += chunk));
    process.stdin.on('end', () => resolve(buf.split(/\r?\n/)));
    process.stdin.on('error', reject);
  });
}

async function main(argv) {
  let args;
  try {
    args = parseArgs(argv);
  } catch (err) {
    console.error(`ERROR: ${err.message}`);
    process.exit(2);
  }

  let lines;
  if (args.input === '-') {
    try {
      lines = await readAllLinesFromStdin();
    } catch (e) {
      console.error(`ERROR: failed to read stdin: ${e.message}`);
      process.exit(2);
    }
  } else {
    try {
      const txt = fs.readFileSync(args.input, 'utf8');
      lines = txt.split(/\r?\n/);
    } catch (e) {
      console.error(`ERROR: cannot open input file '${args.input}': ${e.message}`);
      process.exit(2);
    }
  }

  const { graph, queries } = parseGraphAndQueries(lines, { directed: args.directed });

  const actualQueries = queries.length ? queries
    : (args.src && args.dst) ? [[args.src, args.dst]]
    : null;

  if (!actualQueries) {
    console.error('ERROR: no queries found in input and --from/--to not provided.');
    process.exit(3);
  }

  for (const [src, dst] of actualQueries) {
    const path = bfsShortestPath(graph, src, dst);
    if (!path) {
      console.log(`NO PATH: ${src} -> ${dst}`);
    } else {
      const hops = path.length - 1;
      console.log(`PATH (${hops} edges): ${formatPath(path)}`);
    }
  }
}

// Node-style "entrypoint" check.
if (require.main === module) {
  // Drop the first two argv entries: node, script.js
  main(process.argv.slice(2)).catch(err => {
    console.error(`FATAL: ${err.stack || err.message || err}`);
    process.exit(1);
  });
}

