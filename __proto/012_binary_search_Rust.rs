// main.rs
//
// Binary search over a sorted (ascending) list of numbers, as a tiny CLI.
// - Usage:
//     cargo run -- <target> [path/to/file]
//     # or, without file, read numbers (one per line) from STDIN:
//     printf "1\n3\n4\n7\n9\n11\n15\n" | cargo run -- 11
//
// Behavior
//   • O(log N) classic binary search on f64 values.
//   • Returns the FIRST occurrence if duplicates exist (stable-left).
//   • On hit : prints  FOUND <target> at index <i> (1-based)
//   • On miss: prints  NOT FOUND <target>. Insertion index <i> (1-based), between <left> and <right>
//   • Validates non-decreasing order and errors out if violated.
//
// Notes
//   • If you prefer integers, change f64→i64 and parsing accordingly.

use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::process;

// -----------------------------
// Core algorithm (library API)
// -----------------------------

/// Binary search over ascending slice `arr` for `target`.
/// Returns (idx_1based, found, insert_pos_1based).
fn binary_search_first(arr: &[f64], target: f64) -> (usize, bool, usize) {
    let mut lo: isize = 0;
    let mut hi: isize = arr.len() as isize - 1;
    let mut found_idx: isize = -1;

    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let v = arr[mid as usize];
        if v == target {
            found_idx = mid;
            // keep searching left half to get FIRST occurrence
            hi = mid - 1;
        } else if v < target {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    if found_idx >= 0 {
        (found_idx as usize + 1, true, (lo as usize) + 1)
    } else {
        ((0usize), false, (lo as usize) + 1)
    }
}

/// Ensure the numbers are non-decreasing (ascending, duplicates allowed).
fn ensure_ascending(arr: &[f64]) -> Result<(), String> {
    if arr.len() <= 1 {
        return Ok(());
    }
    let mut last = arr[0];
    for (i, &v) in arr.iter().enumerate().skip(1) {
        if v < last {
            return Err(format!(
                "input is not in ascending order at position {}: {} < {}",
                i + 1,
                v,
                last
            ));
        }
        last = v;
    }
    Ok(())
}

// -----------------------------
// I/O helpers
// -----------------------------

fn read_numbers<R: BufRead>(reader: R) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(1024);
    for (lineno, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| format!("I/O error at line {}: {}", lineno + 1, e))?;
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        match s.parse::<f64>() {
            Ok(v) => out.push(v),
            Err(_) => eprintln!("WARN: skipping non-numeric line {}: {:?}", lineno + 1, s),
        }
    }
    Ok(out)
}

fn read_numbers_from_path<P: AsRef<Path>>(p: P) -> Result<Vec<f64>, String> {
    let file = File::open(p.as_ref())
        .map_err(|e| format!("failed to open {:?}: {}", p.as_ref(), e))?;
    let reader = BufReader::new(file);
    read_numbers(reader)
}

fn usage(prog: &str) {
    eprintln!("Usage:");
    eprintln!("  {prog} <target_number> [path/to/file]");
    eprintln!("  {prog} <target_number>   # read numbers from STDIN (one per line)\n");
    eprintln!("Example:");
    eprintln!("  printf \"1\\n3\\n4\\n7\\n9\\n11\\n15\\n\" | {prog} 11");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let prog = args.get(0).map(String::as_str).unwrap_or("bsearch");

    if args.len() < 2 {
        eprintln!("ERROR: missing <target_number>");
        usage(prog);
        process::exit(2);
    }

    let target: f64 = match args[1].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("ERROR: target_number must be numeric, got {:?}", args[1]);
            usage(prog);
            process::exit(2);
        }
    };

    let numbers: Vec<f64> = if args.len() >= 3 {
        match read_numbers_from_path(&args[2]) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("ERROR: {}", e);
                process::exit(1);
            }
        }
    } else {
        let stdin = io::stdin();
        let lock = stdin.lock();
        match read_numbers(lock) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("ERROR: {}", e);
                process::exit(1);
            }
        }
    };

    if numbers.is_empty() {
        eprintln!("ERROR: no numeric input provided");
        process::exit(1);
    }

    if let Err(e) = ensure_ascending(&numbers) {
        eprintln!("ERROR: {}", e);
        process::exit(1);
    }

    let (idx, found, ins) = binary_search_first(&numbers, target);
    if found {
        println!("FOUND {} at index {} (1-based)", target, idx);
    } else {
        let left = if ins >= 2 {
            format!("{}", numbers[ins - 2])
        } else {
            String::from("-inf")
        };
        let right = if ins - 1 < numbers.len() {
            format!("{}", numbers[ins - 1])
        } else {
            String::from("+inf")
        };
        println!(
            "NOT FOUND {}. Insertion index {} (1-based), between {} and {}",
            target, ins, left, right
        );
    }
}
