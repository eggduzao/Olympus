#!/usr/bin/perl
use strict;
use warnings;

sub levenshtein {
    my ($s1, $s2) = @_;
    my @s1 = split //, $s1;
    my @s2 = split //, $s2;
    my $m = @s1;
    my $n = @s2;

    my @dp;
    for my $i (0..$m) {
        $dp[$i][0] = $i;
    }
    for my $j (0..$n) {
        $dp[0][$j] = $j;
    }

    for my $i (1..$m) {
        for my $j (1..$n) {
            my $cost = ($s1[$i-1] eq $s2[$j-1]) ? 0 : 1;
            my $del = $dp[$i-1][$j] + 1;
            my $ins = $dp[$i][$j-1] + 1;
            my $sub = $dp[$i-1][$j-1] + $cost;
            $dp[$i][$j] = ($del < $ins ? $del : $ins);
            $dp[$i][$j] = ($dp[$i][$j] < $sub ? $dp[$i][$j] : $sub);
        }
    }
    return $dp[$m][$n];
}

# Read input
chomp(my $s1 = <>);
chomp(my $s2 = <>);

print levenshtein($s1, $s2), "\n";

