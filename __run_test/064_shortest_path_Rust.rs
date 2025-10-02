// bfs_unweighted.rs
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
//   rustc bfs_unweighted.rs && ./bfs_unweighted graph.tsv
//   # or pipe:
//   cat graph.tsv | ./bfs_unweighted
//
// OUTPUT
//   One line per query, either "SRC -> ... -> DST" or an explanatory message.

use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::process;

type Adj = HashMap<String, Vec<String>>;
#[derive(Clone)]
struct Query {
    src: String,
    dst: String,
}

fn main() {
    // Input: file path in argv[1] or stdin
    let args: Vec<String> = env::args().collect();
    let reader: Box<dyn BufRead> = if args.len() >= 2 {
        match File::open(&args[1]) {
            Ok(f) => Box::new(BufReader::new(f)),
            Err(e) => {
                eprintln!("Failed to open {}: {}", args[1], e);
                process::exit(1);
            }
        }
    } else {
        Box::new(BufReader::new(io::stdin()))
    };

    let (adj, queries) = parse_input(reader);

    if queries.is_empty() {
        println!(r#"No queries found (expect lines like: "#  SRC  DST"). Nothing to do."#);
        return;
    }

    for q in queries {
        if q.src == q.dst {
            println!("{}", q.src);
            continue;
        }
        if !adj.contains_key(&q.src) {
            println!("No path found from {} to {} (source not in graph)", q.src, q.dst);
            continue;
        }
        if !adj.contains_key(&q.dst) {
            println!("No path found from {} to {} (destination not in graph)", q.src, q.dst);
            continue;
        }
        match bfs_path(&adj, &q.src, &q.dst) {
            Some(path) => println!("{}", path.join(" -> ")),
            None => println!("No path found from {} to {}", q.src, q.dst),
        }
    }
}

//--------------------------- Parsing & Utilities ---------------------------//

// Lines beginning with '#' are queries: "# SRC DST"
// Other lines: "U V1 [V2 ...]" = undirected edges U—V1, U—V2, ...
fn parse_input<R: BufRead>(mut reader: R) -> (Adj, Vec<Query>) {
    let mut adj: Adj = HashMap::new();
    let mut queries: Vec<Query> = Vec::new();

    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf).unwrap_or(0);
        if n == 0 {
            break;
        }
        let line = buf.trim();
        if line.is_empty() {
            continue;
        }
        let mut toks = line.split_whitespace();
        let first = match toks.next() {
            Some(x) => x,
            None => continue,
        };

        if first.starts_with('#') {
            // Query line
            let src = match toks.next() {
                Some(s) => s.to_string(),
                None => continue, // malformed query; skip
            };
            let dst = match toks.next() {
                Some(s) => s.to_string(),
                None => continue,
            };
            queries.push(Query { src, dst });
        } else {
            // Edge list: u v1 v2 ...
            let u = first.to_string();
            let neighbors: Vec<String> = toks.map(|s| s.to_string()).collect();
            if neighbors.is_empty() {
                continue; // malformed; skip
            }
            for v in neighbors {
                add_undirected_edge(&mut adj, &u, &v);
            }
        }
    }

    (adj, queries)
}

fn add_undirected_edge(adj: &mut Adj, u: &str, v: &str) {
    add_directed_unique(adj, u, v);
    add_directed_unique(adj, v, u);
}

fn add_directed_unique(adj: &mut Adj, u: &str, v: &str) {
    let entry = adj.entry(u.to_string()).or_insert_with(Vec::new);
    if !entry.iter().any(|x| x == v) {
        entry.push(v.to_string());
    }
    // Ensure v exists as a key too (even if no outgoing yet), so membership checks work
    adj.entry(v.to_string()).or_insert_with(Vec::new);
}

//--------------------------- BFS Shortest Path -----------------------------//

fn bfs_path(adj: &Adj, src: &str, dst: &str) -> Option<Vec<String>> {
    if src == dst {
        return Some(vec![src.to_string()]);
    }

    let mut visited: HashMap<&str, bool> = HashMap::new();
    let mut parent: HashMap<&str, &str> = HashMap::new();
    let mut q: VecDeque<&str> = VecDeque::new();

    visited.insert(src, true);
    q.push_back(src);

    while let Some(u) = q.pop_front() {
        if let Some(nei) = adj.get(u) {
            for v in nei {
                let v_str: &str = v.as_str();
                if !visited.contains_key(v_str) {
                    visited.insert(v_str, true);
                    parent.insert(v_str, u);
                    if v_str == dst {
                        return Some(reconstruct(&parent, src, dst));
                    }
                    q.push_back(v_str);
                }
            }
        }
    }
    None
}

fn reconstruct(parent: &HashMap<&str, &str>, src: &str, dst: &str) -> Vec<String> {
    let mut path: Vec<String> = Vec::new();
    let mut cur = dst;
    path.push(cur.to_string());
    while cur != src {
        if let Some(&p) = parent.get(cur) {
            cur = p;
            path.push(cur.to_string());
        } else {
            // Shouldn't happen if called correctly; return minimal path
            break;
        }
    }
    path.reverse();
    path
}

