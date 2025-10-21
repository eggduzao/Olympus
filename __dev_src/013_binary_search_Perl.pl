# binary_search.pl
#
# Binary search over a sorted (ascending) list of numbers, as a tiny CLI.
# Usage:
#   perl binary_search.pl <target_number> [path/to/file]
#   # or read numbers (one per line) from STDIN:
#   printf "1\n3\n4\n7\n9\n11\n15\n" | perl binary_search.pl 11
#
# Behavior
#   • O(log N) classic binary search on numeric values (treated as Perl numbers).
#   • Returns the FIRST occurrence if duplicates exist (stable-left).
#   • On hit : prints  FOUND <target> at index <i> (1-based)
#   • On miss: prints  NOT FOUND <target>. Insertion index <i> (1-based), between <left> and <right>
#   • Validates non-decreasing order and errors out if violated.

use strict;
use warnings;
use Scalar::Util qw(looks_like_number);

# -----------------------------
# Core algorithm
# -----------------------------

# Binary search for the first occurrence (stable-left).
# @arr must be ascending (duplicates allowed). $target is numeric.
# Returns (idx_1based, $found_bool, insert_pos_1based).
sub binary_search_first {
    my ($arr_ref, $target) = @_;
    my @a = @$arr_ref;

    my ($lo, $hi) = (0, $#a);
    my $found = -1;

    while ($lo <= $hi) {
        my $mid = $lo + int(($hi - $lo) / 2);
        my $v = $a[$mid];

        if ($v == $target) {
            $found = $mid;
            $hi = $mid - 1;  # keep searching left for first occurrence
        } elsif ($v < $target) {
            $lo = $mid + 1;
        } else {
            $hi = $mid - 1;
        }
    }

    if ($found >= 0) {
        return ($found + 1, 1, $lo + 1);
    } else {
        return (0, 0, $lo + 1);
    }
}

# Ensure ascending (non-decreasing) order; dies with message if violated.
sub ensure_ascending {
    my ($arr_ref) = @_;
    my @a = @$arr_ref;
    return if @a <= 1;

    for (my $i = 1; $i < @a; $i++) {
        if ($a[$i] < $a[$i - 1]) {
            die sprintf("ERROR: input is not in ascending order at position %d: %s < %s\n",
                        $i + 1, $a[$i], $a[$i - 1]);
        }
    }
}

# -----------------------------
# I/O helpers
# -----------------------------

sub read_numbers_from_fh {
    my ($fh) = @_;
    my @out;
    my $lineno = 0;

    while (my $line = <$fh>) {
        $lineno++;
        chomp $line;
        $line =~ s/^\s+|\s+$//g;
        next if $line eq '';

        if (looks_like_number($line)) {
            push @out, 0 + $line; # numeric context
        } else {
            warn sprintf("WARN: skipping non-numeric line %d: %s\n", $lineno, $line);
        }
    }
    return \@out;
}

sub read_numbers_from_path {
    my ($path) = @_;
    open my $fh, '<', $path or die "ERROR: failed to open '$path': $!\n";
    my $nums = read_numbers_from_fh($fh);
    close $fh;
    return $nums;
}

sub usage {
    my ($prog) = @_;
    $prog ||= 'binary_search.pl';
    print STDERR <<"USAGE";
Usage:
  perl $prog <target_number> [path/to/file]
  perl $prog <target_number>            # read numbers from STDIN (one per line)

Example:
  printf "1\\n3\\n4\\n7\\n9\\n11\\n15\\n" | perl $prog 11
USAGE
}

# -----------------------------
# Entrypoint
# -----------------------------

sub main {
    my ($prog, @args) = ($0, @ARGV);

    if (!@args) {
        print STDERR "ERROR: missing <target_number>\n";
        usage($prog);
        return 2;
    }

    my $target_raw = shift @args;
    if (!looks_like_number($target_raw)) {
        print STDERR "ERROR: target_number must be numeric, got '$target_raw'\n";
        usage($prog);
        return 2;
    }
    my $target = 0 + $target_raw;

    my $numbers_ref;
    if (@args) {
        my $path = $args[0];
        $numbers_ref = read_numbers_from_path($path);
    } else {
        $numbers_ref = read_numbers_from_fh(*STDIN);
    }

    if (!@$numbers_ref) {
        print STDERR "ERROR: no numeric input provided\n";
        return 1;
    }

    ensure_ascending($numbers_ref);

    my ($idx1, $found, $ins1) = binary_search_first($numbers_ref, $target);

    if ($found) {
        print "FOUND $target at index $idx1 (1-based)\n";
    } else {
        my $left  = ($ins1 >= 2) ? $numbers_ref->[$ins1 - 2] : '-inf';
        my $right = ($ins1 - 1 < @$numbers_ref) ? $numbers_ref->[$ins1 - 1] : '+inf';
        print "NOT FOUND $target. Insertion index $ins1 (1-based), between $left and $right\n";
    }

    return 0;
}

# Call entrypoint
exit(main());

