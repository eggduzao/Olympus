// quicksort_inplace.rs
// In-place quicksort on an array in Rust.
//
// - If called with a filename, reads one number per line from that file.
// - Otherwise, reads numbers from STDIN (one per line).
// - Skips blank lines and lines beginning with '#'.
// - Prints the sorted numbers (one per line) to STDOUT.
//
// Usage (file):
//   rustc quicksort_inplace.rs && ./quicksort_inplace input.txt
//
// Usage (stdin):
//   printf "5\n3\n8\n1\n2\n" | rustc quicksort_inplace.rs -O && ./quicksort_inplace
//
// Notes:
// - Uses Hoare partition scheme and an explicit stack to avoid deep recursion.
// - Sorts f64 values. Integer-looking values are printed without decimals.

use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    let nums = match args.get(1) {
        Some(path) => read_numbers_from_reader(BufReader::new(
            File::open(path).unwrap_or_else(|e| die(&format!("failed to open {}: {}", path, e))),
        )),
        None => read_numbers_from_reader(BufReader::new(io::stdin())),
    }
    .unwrap_or_else(|e| die(&format!("failed to read numbers: {}", e)));

    let mut nums = nums;
    quicksort_in_place(&mut nums);

    let mut out = io::BufWriter::new(io::stdout());
    for x in nums {
        if x.is_finite() && (x.fract() == 0.0) {
            // Print integers without trailing .0
            writeln!(out, "{:.0}", x).ok();
        } else {
            writeln!(out, "{}", x).ok();
        }
    }
}

/// Print an error message to STDERR and exit(1).
fn die(msg: &str) -> ! {
    eprintln!("{}", msg);
    process::exit(1);
}

/// Read f64 numbers (one per line) skipping blanks and lines starting with '#'.
fn read_numbers_from_reader<R: BufRead>(reader: R) -> io::Result<Vec<f64>> {
    let mut out = Vec::with_capacity(1024);
    for (lineno, line_res) in reader.lines().enumerate() {
        let line = line_res?;
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        match s.parse::<f64>() {
            Ok(v) => out.push(v),
            Err(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: cannot parse {:?} as number", lineno + 1, s),
                ))
            }
        }
    }
    Ok(out)
}

/// In-place quicksort using Hoare partition and an explicit stack.
fn quicksort_in_place(a: &mut [f64]) {
    if a.len() <= 1 {
        return;
    }

    // We'll manage segments [lo, hi] inclusive.
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(64);
    stack.push((0, a.len() - 1));

    while let Some((mut lo, mut hi)) = stack.pop() {
        while lo < hi {
            let p = hoare_partition(a, lo, hi);

            // Tail-call elimination: process smaller side now, push larger side.
            let left_len = p.saturating_sub(lo) + 1; // [lo..=p]
            let right_len = hi.saturating_sub(p + 1) + 1; // [p+1..=hi], careful with underflow

            if left_len < right_len {
                // Push right, loop on left
                if p + 1 <= hi {
                    stack.push((p + 1, hi));
                }
                hi = p;
            } else {
                // Push left, loop on right
                if lo <= p {
                    stack.push((lo, p));
                }
                lo = p + 1;
            }
        }
    }
}

/// Hoare partition on a[lo..=hi]. Returns index p such that
/// all elements in a[lo..=p] <= pivot and a[p+1..=hi] >= pivot.
fn hoare_partition(a: &mut [f64], lo: usize, hi: usize) -> usize {
    let pivot = a[lo + (hi - lo) / 2];
    let mut i = lo as isize - 1;
    let mut j = hi as isize + 1;

    loop {
        // Move i right while a[i] < pivot
        loop {
            i += 1;
            if !(a[i as usize] < pivot) {
                break;
            }
        }
        // Move j left while a[j] > pivot
        loop {
            j -= 1;
            if !(a[j as usize] > pivot) {
                break;
            }
        }
        if i >= j {
            return j as usize;
        }
        a.swap(i as usize, j as usize);
    }
}

