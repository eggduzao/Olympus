'use strict';
/**
 * In-place quicksort on an array (iterative, Hoare partition, median-of-three).
 *
 * Usage
 * -----
 * # stdin (one number per line)
 * node quicksort_inplace.js < numbers.txt
 *
 * # or pass a file path
 * node quicksort_inplace.js numbers.txt
 *
 * Input format
 * ------------
 * - One number per line (integer or float).
 * - Blank lines are ignored.
 * - Lines starting with '#' are comments.
 *
 * Output
 * ------
 * - Sorted numbers, one per line.
 */

const fs = require('fs');

// ---------- I/O ----------
function parseNumbers(text) {
  const out = [];
  const lines = text.split(/\r?\n/);
  for (let raw of lines) {
    const s = raw.trim();
    if (!s || s.startsWith('#')) continue;
    let n;
    if (/[.eE]/.test(s)) {
      n = Number.parseFloat(s);
    } else {
      n = Number.parseInt(s, 10);
    }
    if (!Number.isFinite(n)) {
      throw new Error(`Invalid numeric line: ${JSON.stringify(raw)}`);
    }
    out.push(n);
  }
  return out;
}

function printNumbers(arr) {
  process.stdout.write(arr.map(x => String(x)).join('\n') + '\n');
}

// ---------- Sorting primitives ----------
function medianOfThree(a, i, j, k) {
  const ai = a[i], aj = a[j], ak = a[k];
  if (ai < aj) {
    if (aj < ak) return j;        // ai < aj < ak
    return ai < ak ? i : k;       // ai < ak <= aj  OR  ak <= ai < aj
  } else {
    if (ai < ak) return i;        // aj <= ai < ak
    return aj < ak ? j : k;       // aj < ak <= ai  OR  ak <= aj <= ai
  }
}

function insertionSort(a, lo, hi) {
  for (let i = lo + 1; i <= hi; i++) {
    const key = a[i];
    let j = i - 1;
    while (j >= lo && a[j] > key) {
      a[j + 1] = a[j];
      j--;
    }
    a[j + 1] = key;
  }
}

function hoarePartition(a, lo, hi) {
  const mid = lo + ((hi - lo) >> 1);
  const pidx = medianOfThree(a, lo, mid, hi);
  const pivot = a[pidx];

  let i = lo - 1;
  let j = hi + 1;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    do { i++; } while (a[i] < pivot);
    do { j--; } while (a[j] > pivot);
    if (i >= j) return j;
    // swap
    const tmp = a[i]; a[i] = a[j]; a[j] = tmp;
  }
}

function quicksortInPlace(a) {
  const n = a.length;
  if (n < 2) return;

  const SMALL = 16; // cutoff for insertion sort
  const stack = [[0, n - 1]]; // inclusive ranges

  while (stack.length) {
    const [lo, hi] = stack.pop();

    if (hi - lo + 1 <= SMALL) {
      insertionSort(a, lo, hi);
      continue;
    }

    const p = hoarePartition(a, lo, hi);

    const leftSize  = p - lo + 1;
    const rightSize = hi - (p + 1) + 1;

    // Process smaller side next to keep stack shallow
    if (leftSize < rightSize) {
      if (lo < p) stack.push([p + 1, hi]), stack.push([lo, p]);
      else stack.push([p + 1, hi]);
    } else {
      if (p + 1 < hi) stack.push([lo, p]), stack.push([p + 1, hi]);
      else stack.push([lo, p]);
    }
  }
}

// ---------- Main ----------
function main(argv) {
  let text;
  if (argv.length > 2) {
    // file path provided
    text = fs.readFileSync(argv[2], 'utf8');
  } else {
    // read entire stdin
    text = fs.readFileSync(0, 'utf8');
  }

  const arr = parseNumbers(text);
  quicksortInPlace(arr);
  printNumbers(arr);
}

if (require.main === module) {
  try {
    main(process.argv);
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  }
}

