<?php
/**
 * InPlaceQuicksort.php
 *
 * In-place quicksort on an array in PHP.
 *
 * • Reads integers (one per line) from a file passed as the first CLI arg,
 *   or from STDIN if no file is given.
 * • Skips blank lines and lines starting with '#'.
 * • Uses an iterative quicksort (manual stack) with Hoare partition and
 *   median-of-three pivoting to reduce worst cases.
 * • Prints sorted numbers, one per line, to STDOUT.
 *
 * Run:
 *   php InPlaceQuicksort.php input.txt
 *   # or
 *   printf "5\n3\n8\n1\n2\n" | php InPlaceQuicksort.php
 */

declare(strict_types=1);

if (PHP_SAPI !== 'cli') {
    fwrite(STDERR, "Run this script from the command line.\n");
    exit(1);
}

// ---------- Input ----------
$lines = [];
if ($argc > 1) {
    $fp = @fopen($argv[1], 'r');
    if ($fp === false) {
        fwrite(STDERR, "Failed to open file: {$argv[1]}\n");
        exit(1);
    }
    while (($line = fgets($fp)) !== false) {
        $lines[] = $line;
    }
    fclose($fp);
} else {
    // Read all from STDIN
    while (($line = fgets(STDIN)) !== false) {
        $lines[] = $line;
    }
}

// Parse integers; skip blanks and comments
$xs = [];
foreach ($lines as $idx => $raw) {
    $s = trim($raw);
    if ($s === '' || ($s[0] ?? '') === '#') {
        continue;
    }
    // accept leading +/-, digits; fail otherwise
    if (!preg_match('/^[+-]?\d+$/', $s)) {
        fwrite(STDERR, "Line ".($idx+1)." is not an integer: {$s}\n");
        exit(1);
    }
    // Use intdiv-safe range: PHP ints are platform-dependent but usually 64-bit
    $xs[] = (int)$s;
}

$n = count($xs);
if ($n <= 1) {
    foreach ($xs as $v) {
        echo $v, PHP_EOL;
    }
    exit(0);
}

// ---------- Quicksort (in place) ----------
/**
 * Iterative quicksort using Hoare partition scheme.
 * Sorts the array of integers in ascending order.
 *
 * @param array<int,int> &$a
 * @return void
 */
function quicksort_in_place(array &$a): void
{
    $lo = 0;
    $hi = count($a) - 1;

    // Manual stack of ranges [l, h]
    /** @var SplStack<array{0:int,1:int}> $stack */
    $stack = new SplStack();
    $stack->push([$lo, $hi]);

    while (!$stack->isEmpty()) {
        [$l, $h] = $stack->pop();
        if ($l >= $h) {
            continue;
        }

        $p = hoare_partition($a, $l, $h);

        // Sort smaller side first to keep the stack shallow
        $leftSize  = $p - $l + 1;
        $rightSize = $h - ($p + 1) + 1;

        if ($leftSize < $rightSize) {
            if ($p + 1 < $h) $stack->push([$p + 1, $h]);
            if ($l < $p)     $stack->push([$l, $p]);
        } else {
            if ($l < $p)     $stack->push([$l, $p]);
            if ($p + 1 < $h) $stack->push([$p + 1, $h]);
        }
    }
}

/**
 * Hoare partition with median-of-three pivot selection (value-based).
 * Returns index j such that all elements in [l, j] <= pivot and all in [j+1, h] >= pivot.
 *
 * @param array<int,int> &$a
 * @param int $l
 * @param int $h
 * @return int
 */
function hoare_partition(array &$a, int $l, int $h): int
{
    $m = $l + intdiv($h - $l, 2);
    $pivot = median3($a[$l], $a[$m], $a[$h]);

    $i = $l - 1;
    $j = $h + 1;

    while (true) {
        // Move i right while a[i] < pivot
        do {
            $i++;
        } while ($a[$i] < $pivot);

        // Move j left while a[j] > pivot
        do {
            $j--;
        } while ($a[$j] > $pivot);

        if ($i >= $j) {
            return $j;
        }
        // swap
        $tmp = $a[$i];
        $a[$i] = $a[$j];
        $a[$j] = $tmp;
    }
}

/**
 * Median of three values.
 *
 * @template T of int|float
 * @param T $x
 * @param T $y
 * @param T $z
 * @return T
 */
function median3($x, $y, $z)
{
    if (($x <= $y && $y <= $z) || ($z <= $y && $y <= $x)) return $y;
    if (($y <= $x && $x <= $z) || ($z <= $x && $x <= $y)) return $x;
    return $z;
}

// Sort and emit
quicksort_in_place($xs);
foreach ($xs as $v) {
    echo $v, PHP_EOL;
}

