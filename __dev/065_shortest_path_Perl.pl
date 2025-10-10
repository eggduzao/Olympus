# bfs_unweighted.pl
# Breadth-First Search (BFS) shortest path on an unweighted, *undirected* graph.
#
# INPUT FORMAT (tab- or space-separated; one record per line):
#   - Edge lines: "U  V1 [V2 ...]" meaning undirected edges U—V1, U—V2, ...
#   - Query lines: start with "#" then "SRC  DST"
#
# Example:
#   A   B   F
#   B   A   C
#   C   B   D
#   D   C   E
#   E   D   F
#   F   A   E
#   #   A   E
#
# USAGE
#   perl bfs_unweighted.pl graph.tsv
#   # or:
#   cat graph.tsv | perl bfs_unweighted.pl
#
# OUTPUT
#   One line per query: either "SRC -> ... -> DST" or an explanatory message.

use strict;
use warnings;
use IO::Handle;

STDOUT->autoflush(1);
STDERR->autoflush(1);

sub main {
    my ($fh, $src_name) = input_handle();
    my ($adj, $queries) = parse_input($fh);

    if (!@$queries) {
        print "No queries found (expect lines like: \"# SRC DST\"). Nothing to do.\n";
        return;
    }

    for my $q (@$queries) {
        my ($src, $dst) = @$q{qw/src dst/};

        if ($src eq $dst) {
            print "$src\n";
            next;
        }
        if (!exists $adj->{$src}) {
            print "No path found from $src to $dst (source not in graph)\n";
            next;
        }
        if (!exists $adj->{$dst}) {
            print "No path found from $src to $dst (destination not in graph)\n";
            next;
        }

        my $path = bfs_path($adj, $src, $dst);
        if ($path) {
            print join(" -> ", @$path), "\n";
        } else {
            print "No path found from $src to $dst\n";
        }
    }
}

#------------------------------- I/O ---------------------------------------#

sub input_handle {
    if (@ARGV >= 1) {
        my $file = $ARGV[0];
        open my $fh, "<", $file or die "Failed to open $file: $!";
        return ($fh, $file);
    } else {
        return (*STDIN{IO}, "STDIN");
    }
}

# Parse lines into:
#   - $adj: { node => [unique neighbors...] }
#   - $queries: [ {src => "...", dst => "..."}, ... ]
sub parse_input {
    my ($fh) = @_;
    my %adj;
    my @queries;

    while (defined(my $line = <$fh>)) {
        chomp $line;
        $line =~ s/^\s+|\s+$//g;        # trim
        next if $line eq '';
        my @tok = split /\s+/, $line;
        next unless @tok;

        if ($tok[0] =~ /^\#/) {
            # Query: "# SRC DST" (ignore additional tokens)
            my $src = $tok[1] // next;
            my $dst = $tok[2] // next;
            push @queries, { src => $src, dst => $dst };
        } else {
            # Edges: "U V1 [V2 ...]" => undirected U—Vi for all i
            my $u = shift @tok;
            next unless @tok;
            for my $v (@tok) {
                add_undirected_edge(\%adj, $u, $v);
            }
        }
    }

    return (\%adj, \@queries);
}

sub add_undirected_edge {
    my ($adj, $u, $v) = @_;
    add_directed_unique($adj, $u, $v);
    add_directed_unique($adj, $v, $u);
    # ensure both appear as keys
    $adj->{$u} //= [];
    $adj->{$v} //= [];
}

sub add_directed_unique {
    my ($adj, $u, $v) = @_;
    $adj->{$u} //= [];
    # de-duplicate cheaply with a small seen hash per insertion pass
    # (avoid O(deg) linear scan for massively repeated edges)
    state %seen_tmp;
    my $key = "$u\0$v";
    unless ($seen_tmp{$key}++) {
        # still ensure uniqueness against what's already there
        my $exists = 0;
        for my $x (@{$adj->{$u}}) {
            if ($x eq $v) { $exists = 1; last; }
        }
        push @{$adj->{$u}}, $v unless $exists;
    }
}

#------------------------------- BFS ---------------------------------------#

# Return shortest path [src, ..., dst] or undef
sub bfs_path {
    my ($adj, $src, $dst) = @_;

    # For memory efficiency on huge graphs, keep visitation as a hash of flags,
    # parent as { child => parent }, and use an array + head index as queue.
    my %visited;
    my %parent;
    my @queue;
    my $head = 0;

    $visited{$src} = 1;
    push @queue, $src;

    while ($head < @queue) {
        my $u = $queue[$head++];
        my $nbrs = $adj->{$u} // [];
        for my $v (@$nbrs) {
            next if $visited{$v};
            $visited{$v} = 1;
            $parent{$v} = $u;
            if ($v eq $dst) {
                return reconstruct_path(\%parent, $src, $dst);
            }
            push @queue, $v;
        }
    }
    return undef;
}

sub reconstruct_path {
    my ($parent, $src, $dst) = @_;
    my @path;
    my $cur = $dst;
    push @path, $cur;
    while ($cur ne $src) {
        my $p = $parent->{$cur};
        last unless defined $p; # safety guard for malformed call
        $cur = $p;
        push @path, $cur;
    }
    @path = reverse @path;
    return \@path;
}

main();

