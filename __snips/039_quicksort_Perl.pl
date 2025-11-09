# quicksort_inplace.pl
# In-place quicksort on an array in Perl.
#
# - If called with a filename, reads one number per line from that file.
# - Otherwise, reads numbers from STDIN (one per line).
# - Skips blank lines and lines beginning with '#'.
# - Prints the sorted numbers (one per line) to STDOUT.
#
# Run:
#   perl quicksort_inplace.pl input.txt
#   # or
#   printf "5\n3\n8\n1\n2\n" | perl quicksort_inplace.pl

use strict;
use warnings;
use Scalar::Util qw(looks_like_number);

# ---------- Main ----------
my @nums = @ARGV ? read_numbers($ARGV[0]) : read_numbers();
quicksort_inplace(\@nums);

for my $x (@nums) {
    if (defined $x && $x == int($x)) {
        print int($x), "\n";    # print integers without trailing .0
    } else {
        print $x, "\n";
    }
}

# ---------- IO ----------
sub read_numbers {
    my ($path) = @_;
    my $fh;
    if (defined $path) {
        open $fh, '<', $path or die "failed to open $path: $!";
    } else {
        $fh = *STDIN;
    }

    my @out;
    my $lineno = 0;
    while (my $line = <$fh>) {
        ++$lineno;
        $line =~ s/^\s+|\s+$//g;            # trim
        next if $line eq '' || $line =~ /^#/;
        die "line $lineno: not a number: $line\n"
            unless looks_like_number($line);
        push @out, 0 + $line;               # coerce to numeric
    }
    close $fh if defined $path;
    return @out;
}

# ---------- Quicksort (in-place, iterative, Hoare partition) ----------
sub quicksort_inplace {
    my ($a) = @_;
    return if @$a < 2;

    # Work on inclusive segments [lo, hi]. Use an explicit stack to avoid deep recursion.
    my @stack = (0, $#$a);

    while (@stack) {
        my $hi = pop @stack;
        my $lo = pop @stack;
        next if $lo >= $hi;

        my $p = hoare_partition($a, $lo, $hi);

        # Process the smaller side first (tail-call elimination style).
        my $left_len  = $p - $lo + 1;       # [lo .. p]
        my $right_len = $hi - ($p + 1) + 1; # [p+1 .. hi]

        if ($left_len < $right_len) {
            push @stack, $p + 1, $hi;       # sort right later
            push @stack, $lo, $p;           # continue with left
        } else {
            push @stack, $lo, $p;           # sort left later
            push @stack, $p + 1, $hi;       # continue with right
        }
    }
}

sub hoare_partition {
    my ($a, $lo, $hi) = @_;
    my $pivot = $a->[$lo + int(($hi - $lo) / 2)];
    my $i = $lo - 1;
    my $j = $hi + 1;

    while (1) {
        do { $i++ } while $a->[$i] < $pivot;
        do { $j-- } while $a->[$j] > $pivot;
        return $j if $i >= $j;
        @$a[$i, $j] = @$a[$j, $i];  # swap
    }
}

