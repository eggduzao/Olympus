<?php
/**
 * binsearch.php — Binary search (first occurrence) on a sorted ascending list.
 *
 * Run:
 *   php binsearch.php <target> [path/to/file]
 *   # or via STDIN
 *   printf "1\n3\n4\n7\n9\n11\n15\n" | php binsearch.php 11
 *
 * Behavior:
 *   • O(log N) binary search over an in-memory array.
 *   • Returns FIRST index if duplicates exist (stable-left).
 *   • On hit : prints  FOUND <target> at index <i> (1-based)
 *   • On miss: prints  NOT FOUND <target>. Insertion index <i> (1-based), between <left> and <right>
 *   • Validates non-decreasing order; exits non-zero if violated/no data.
 */

declare(strict_types=1);

// ----------------- Helpers -----------------

/** Print to STDERR. */
function eprintln(string $s): void { fwrite(STDERR, $s . PHP_EOL); }

/** Trim and attempt to parse an integer; return null on failure. */
function parse_int_or_null(string $line): ?int {
    $s = trim($line);
    if ($s === '') return null;
    // allow +/-, decimal only
    if (!preg_match('/^[+-]?\d+$/', $s)) return null;
    // fit into PHP int (platform dependent, but fine for typical datasets)
    return (int)$s;
}

/** Read numbers from a file path or STDIN; warn on bad lines. */
function read_numbers(?string $path): array {
    $lines = [];
    if ($path !== null) {
        if (!is_readable($path)) {
            eprintln("ERROR: cannot read file '$path'");
            exit(2);
        }
        $lines = preg_split("/\r\n|\n|\r/", file_get_contents($path));
    } else {
        $stdin = stream_get_contents(STDIN);
        $lines = preg_split("/\r\n|\n|\r/", $stdin);
    }

    $out = [];
    foreach ($lines as $i => $raw) {
        if ($raw === null) continue;
        $v = parse_int_or_null($raw);
        if ($v === null && trim((string)$raw) !== '') {
            eprintln("WARN: skipping non-integer line " . ($i + 1) . ": " . trim((string)$raw));
        } elseif ($v !== null) {
            $out[] = $v;
        }
    }
    return $out;
}

/** Ensure non-decreasing order; return true if ok, else print error and exit. */
function ensure_ascending(array $xs): void {
    $n = count($xs);
    for ($i = 1; $i < $n; $i++) {
        if ($xs[$i] < $xs[$i - 1]) {
            eprintln("ERROR: input not in ascending order at position " . ($i + 1) .
                     ": {$xs[$i]} < {$xs[$i - 1]}");
            exit(3);
        }
    }
}

/**
 * Lower-bound binary search (first index with value >= target).
 * Returns [foundIndex1Based, foundBool, insertionIndex1Based].
 */
function binary_search_first(array $arr, int $target): array {
    $n = count($arr);
    $lo = 0; $hi = $n; // search in [lo, hi)
    while ($lo < $hi) {
        $mid = intdiv($lo + $hi, 2);
        if ($arr[$mid] < $target) $lo = $mid + 1;
        else $hi = $mid;
    }
    $insertion = $lo; // 0..n
    if ($insertion < $n && $arr[$insertion] === $target) {
        return [$insertion + 1, true, $insertion + 1];
    }
    return [0, false, $insertion + 1];
}

// ----------------- Main -----------------

// Args: <target> [file]
if ($argc < 2) {
    eprintln("Usage: php binsearch.php <target> [path/to/file]");
    exit(1);
}
$targetRaw = $argv[1];
if (!preg_match('/^[+-]?\d+$/', $targetRaw)) {
    eprintln("ERROR: target must be an integer, got '$targetRaw'");
    exit(1);
}
$target = (int)$targetRaw;
$path = $argv[2] ?? null;

// Load & validate data
$nums = read_numbers($path);
if (count($nums) === 0) {
    eprintln("ERROR: no numeric input provided.");
    exit(2);
}
ensure_ascending($nums);

// Search
[$idx1, $found, $ins1] = binary_search_first($nums, $target);

// Context for pretty message
$n = count($nums);
$left  = ($ins1 >= 2)      ? (string)$nums[$ins1 - 2] : "-inf";
$right = ($ins1 - 1 < $n)  ? (string)$nums[$ins1 - 1] : "+inf";

// Report
if ($found) {
    echo "FOUND {$target} at index {$idx1} (1-based)" . PHP_EOL;
} else {
    echo "NOT FOUND {$target}. Insertion index {$ins1} (1-based), between {$left} and {$right}" . PHP_EOL;
}

