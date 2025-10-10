/**
 * Binary search in a sorted array — Node.js script version
 *
 * Usage:
 *   node binary_search.js input.txt 1 9 15 2
 *     - input.txt: text file with one integer per line (unsorted OK; we sort)
 *     - following args are targets to query (if none given, uses a small demo set)
 *
 * Notes:
 *   • Time: O(log n) per query after O(n log n) sort.
 *   • Space: O(n) to hold the array (binary search requires random access).
 *   • For truly huge inputs that don't fit in memory, binary search is not applicable
 *     without an index or an on-disk sorted structure (e.g., SQLite/Parquet + index).
 */

const fs = require("fs");
const readline = require("readline");

/**
 * Binary search over a sorted numeric array.
 * @param {number[]} arr Sorted array (ascending).
 * @param {number} target Value to find.
 * @returns {number|null} Index of found element or null if not found.
 */
function binarySearch(arr, target) {
  let lo = 0, hi = arr.length - 1;
  while (lo <= hi) {
    // Avoid overflow in other languages; fine here but keep the pattern:
    const mid = lo + ((hi - lo) >> 1);
    const v = arr[mid];

    if (v === target) return mid;
    if (v < target) lo = mid + 1;
    else hi = mid - 1;
  }
  return null;
}

/**
 * Stream integers from file (one per line), parse, and return a sorted array.
 * Lines that are empty or fail to parse as integers are skipped with a warning.
 * @param {string} path
 * @returns {Promise<number[]>}
 */
async function loadArrayFromFile(path) {
  if (!fs.existsSync(path)) {
    throw new Error(`Input file not found: ${path}`);
  }
  const stream = fs.createReadStream(path, { encoding: "utf-8" });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

  const out = [];
  for await (const raw of rl) {
    const line = raw.trim();
    if (!line) continue;
    // Allow numbers like "11", "+11", "-11"
    // If you need BigInt, adjust accordingly.
    const n = Number(line);
    if (Number.isFinite(n) && Number.isInteger(n)) {
      out.push(n);
    } else {
      // Non-fatal: skip malformed lines
      console.warn(`[skip] not an integer: ${JSON.stringify(line)}`);
    }
  }
  // Numeric sort (default JS sort is lexicographic)
  out.sort((a, b) => a - b);
  return out;
}

/**
 * Pretty-print the result of a query.
 */
function report(target, idx) {
  if (idx === null) {
    console.log(`Target ${target}: not found`);
  } else {
    console.log(`Target ${target}: found at index ${idx}`);
  }
}

async function main() {
  const [, , inputPath, ...rawTargets] = process.argv;

  if (!inputPath) {
    console.error("Usage: node binary_search.js <input.txt> [targets ...]");
    console.error("Example: node binary_search.js input.txt 1 9 15 2");
    process.exit(2);
  }

  const arr = await loadArrayFromFile(inputPath);
  console.log(`Loaded ${arr.length} integers from ${inputPath}`);

  // If no targets provided, demo with a small set
  const targets = rawTargets.length
    ? rawTargets.map((t) => Number(t)).filter((x) => Number.isFinite(x))
    : [1, 9, 15, 2];

  // Perform queries
  for (const t of targets) {
    const idx = binarySearch(arr, t);
    report(t, idx);
  }
}

// Conventional Node.js script entrypoint
if (require.main === module) {
  main().catch((err) => {
    console.error(err && err.stack ? err.stack : String(err));
    process.exit(1);
  });
}
