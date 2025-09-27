#!/usr/bin/env bash
# BFS shortest path on an unweighted graph (Bash 4+)
# - Reads a simple TSV format from a file path or stdin:
#     NODE<TAB>NEI1
#     NODE<TAB>NEI1<TAB>NEI2          # optional third column
#     #<TAB>SRC<TAB>DST               # one or more queries
# - Treats the graph as UNDIRECTED.
# - Prints the shortest path (fewest hops) as "SRC -> ... -> DST" or a message if unreachable.

set -euo pipefail

########## helpers ##########
die() { echo "ERROR: $*" >&2; exit 1; }

# Append neighbor v to adjacency of u if not already present
add_edge() {
  local u="$1" v="$2"
  [[ -z "$u" || -z "$v" ]] && return 0
  local key="adj[$u]"
  local current="${adj[$u]-}"
  # ensure space-delimited list has unique entries
  if [[ " $current " != *" $v "* ]]; then
    adj["$u"]="${current:+$current }$v"
  fi
}

# Reconstruct path from parents map, printing "a -> b -> ... -> z"
print_path() {
  local src="$1" dst="$2"
  local node="$dst"
  local path=()
  while :; do
    path=("$node" "${path[@]}")
    [[ "$node" == "$src" ]] && break
    local p="${parent[$node]-}"
    if [[ -z "$p" ]]; then
      echo "No path found from $src to $dst"
      return 0
    fi
    node="$p"
  done
  local IFS=" -> "
  echo "${path[*]}"
}

########## input parsing ##########
declare -A adj           # adjacency list: node -> "nei1 nei2 ..."
declare -a queries_src=()
declare -a queries_dst=()

# Read from file path or stdin
input="${1:-/dev/stdin}"
[[ -r "$input" ]] || die "Cannot read input: $input"

# Use awk to normalize to three columns (node, nei1, nei2) or query rows
# Then parse line-by-line in bash to build the graph.
while IFS=$'\t' read -r c1 c2 c3; do
  # skip empty lines / whitespace-only
  [[ -z "${c1//[$' \t\r\n']/}" ]] && continue

  if [[ "$c1" == \#* ]]; then
    # Query line: "# <src> <dst>"
    [[ -z "${c2-}" || -z "${c3-}" ]] && die "Bad query line (need '#\\tSRC\\tDST')"
    queries_src+=("$c2")
    queries_dst+=("$c3")
  else
    # Edge line(s). Add undirected edges.
    # Format supports 2 or 3 columns: NODE  NEI1 [NEI2]
    [[ -n "${c2-}" ]] && { add_edge "$c1" "$c2"; add_edge "$c2" "$c1"; }
    [[ -n "${c3-}" ]] && { add_edge "$c1" "$c3"; add_edge "$c3" "$c1"; }
  fi
done < <(
  # Normalize input: collapse multiple spaces to tabs, keep tabs as tabs
  # Accept both tab- and space-separated inputs.
  # Remove Windows CRs just in case.
  sed -e 's/\r$//' "$input" \
  | awk '
      BEGIN{FS="[ \t]+"; OFS="\t"}
      NF==0{next}
      $1 ~ /^#/ {
        if(NF<3){ printf("##\tBAD\tBAD\n"); next }
        print "#",$2,$3; next
      }
      {
        # Keep up to 3 columns: NODE NEI1 [NEI2]
        if(NF>=2) {
          if(NF>=3) print $1,$2,$3;
          else print $1,$2,"";
        }
      }
    '
)

# If no explicit queries, let the user know and exit gracefully.
if (( ${#queries_src[@]} == 0 )); then
  echo "No queries found (lines starting with '#\\tSRC\\tDST'). Nothing to do."
  exit 0
fi

########## BFS per query ##########
for ((qi=0; qi<${#queries_src[@]}; ++qi)); do
  src="${queries_src[$qi]}"
  dst="${queries_dst[$qi]}"

  # Sanity checks
  if [[ -z "${adj[$src]-}" ]]; then
    echo "No path found from $src to $dst (source '$src' is not in graph)"; continue
  fi
  if [[ -z "${adj[$dst]-}" && "$src" != "$dst" ]]; then
    echo "No path found from $src to $dst (destination '$dst' is not in graph)"; continue
  fi

  # Edge case: src == dst
  if [[ "$src" == "$dst" ]]; then
    echo "$src"; continue
  fi

  # BFS state
  declare -A seen=()
  declare -A parent=()
  declare -a queue=()
  head=0

  # enqueue src
  queue+=("$src")
  seen["$src"]=1
  parent["$src"]=""

  found=0
  # BFS loop
  while (( head < ${#queue[@]} )); do
    node="${queue[$head]}"; ((head++))
    # Early exit if reached dst
    if [[ "$node" == "$dst" ]]; then
      found=1
      break
    fi
    # Expand neighbors
    for nei in ${adj[$node]-}; do
      [[ -n "${seen[$nei]-}" ]] && continue
      seen["$nei"]=1
      parent["$nei"]="$node"
      queue+=("$nei")
    done
  done

  if (( found )); then
    print_path "$src" "$dst"
  else
    echo "No path found from $src to $dst"
  fi

  # cleanup assoc arrays before next query
  unset seen parent queue head found node nei src dst
done

