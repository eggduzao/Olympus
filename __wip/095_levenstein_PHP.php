<?php
function levenshteinDistance($s1, $s2) {
    $m = strlen($s1);
    $n = strlen($s2);

    // Initialize matrix
    $dp = array();
    for ($i = 0; $i <= $m; $i++) {
        $dp[$i] = array();
        for ($j = 0; $j <= $n; $j++) {
            if ($i == 0) {
                $dp[$i][$j] = $j;
            } elseif ($j == 0) {
                $dp[$i][$j] = $i;
            } else {
                $dp[$i][$j] = 0;
            }
        }
    }

    // Fill matrix
    for ($i = 1; $i <= $m; $i++) {
        for ($j = 1; $j <= $n; $j++) {
            $cost = ($s1[$i - 1] == $s2[$j - 1]) ? 0 : 1;
            $dp[$i][$j] = min(
                $dp[$i - 1][$j] + 1,     // Deletion
                $dp[$i][$j - 1] + 1,     // Insertion
                $dp[$i - 1][$j - 1] + $cost // Substitution
            );
        }
    }

    return $dp[$m][$n];
}

// Sample input
$input1 = trim(fgets(STDIN));
$input2 = trim(fgets(STDIN));

echo levenshteinDistance($input1, $input2) . PHP_EOL;
?>

