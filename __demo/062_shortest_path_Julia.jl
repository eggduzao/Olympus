# bfs_unweighted.jl
# Breadth-first search (BFS) shortest path on an unweighted, undirected graph.
#
# INPUT FORMAT (tab- or space-separated; one edge per line; queries start with '#'):
#   A   B   F
#   B   A   C
#   C   B   D
#   D   C   E
#   E   D   F
#   F   A   E
#   #   A   E
#
# USAGE
#   julia bfs_unweighted.jl graph.tsv
#   # or pipe the contents:
#   cat graph.tsv | julia bfs_unweighted.jl
#
# OUTPUT
#   One line per query, either "SRC -> ... -> DST" or an explanatory message.

#--------------------------- Parsing & Utilities ---------------------------#

"""
    parse_input(lines) -> (adj, queries)

Parse lines into:
- `adj::Dict{String,Vector{String}}`: undirected adjacency list
- `queries::Vector{Tuple{String,String}}`: (src, dst) per query line starting with '#'
Each non-query line is `U V1 [V2 ...]` meaning edges U—V1 and (optionally) U—V2.
"""
function parse_input(lines::Vector{String})
    adj     = Dict{String,Vector{String}}()
    queries = Vector{Tuple{String,String}}()

    for (i, raw) in pairs(lines)
        line = strip(raw)
        isempty(line) && continue
        toks = split(line)  # split on any whitespace (spaces or tabs)

        if startswith(toks[1], "#")
            if length(toks) < 3
                @warn "Skipping malformed query line $i: $line"
                continue
            end
            push!(queries, (toks[2], toks[3]))
        else
            if length(toks) < 2
                @warn "Skipping malformed edge line $i: $line"
                continue
            end
            u = toks[1]
            for v in Iterators.drop(toks, 1)
                add_undirected_edge!(adj, u, v)
            end
        end
    end

    return adj, queries
end

"""
    add_undirected_edge!(adj, u, v)

Add u<->v to the adjacency list, ensuring no duplicate neighbors.
"""
function add_undirected_edge!(adj::Dict{String,Vector{String}}, u::String, v::String)
    add_directed_unique!(adj, u, v)
    add_directed_unique!(adj, v, u)
    return adj
end

function add_directed_unique!(adj::Dict{String,Vector{String}}, u::String, v::String)
    if !haskey(adj, u)
        adj[u] = [v]
        return
    end
    nb = adj[u]
    if all(x -> x != v, nb)
        push!(nb, v)
    end
end

#--------------------------- BFS Shortest Path ----------------------------#

"""
    bfs_path(adj, src, dst) -> Vector{String}

Return the shortest path from `src` to `dst` as a vector of node labels,
or an empty vector if no path exists. Graph is unweighted & undirected.
"""
function bfs_path(adj::Dict{String,Vector{String}}, src::String, dst::String)
    src == dst && return [src]
    haskey(adj, src) || return String[]
    haskey(adj, dst) || return String[]

    visited = Set{String}([src])
    parent  = Dict{String,String}()   # child -> parent
    q = Base.Collections.Queue{String}()
    enqueue!(q, src)

    while !isempty(q)
        u = dequeue!(q)
        for v in get(adj, u, String[])
            if v ∉ visited
                push!(visited, v)
                parent[v] = u
                v == dst && return reconstruct_path(parent, src, dst)
                enqueue!(q, v)
            end
        end
    end

    return String[]  # not found
end

function reconstruct_path(parent::Dict{String,String}, src::String, dst::String)
    path = String[dst]
    cur = dst
    while cur != src
        if !haskey(parent, cur)
            return String[]  # safety guard
        end
        cur = parent[cur]
        pushfirst!(path, cur)
    end
    return path
end

#------------------------------ Entrypoint --------------------------------#

function run(io::IO)
    raw = read(io, String)
    # normalize newlines to '\n'
    raw = replace(raw, r"\r\n?" => "\n")
    lines = split(raw, '\n'; keepempty=false)

    adj, queries = parse_input(lines)

    if isempty(queries)
        println("No queries found (expect lines like: \"#\\tSRC\\tDST\"). Nothing to do.")
        return
    end

    for (src, dst) in queries
        if src == dst
            println(src)
            continue
        end
        if !haskey(adj, src)
            println("No path found from $src to $dst (source not in graph)")
            continue
        end
        if !haskey(adj, dst)
            println("No path found from $src to $dst (destination not in graph)")
            continue
        end
        path = bfs_path(adj, src, dst)
        if isempty(path)
            println("No path found from $src to $dst")
        else
            println(join(path, " -> "))
        end
    end
end

# If a filename arg is provided, read it; else read from STDIN.
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1
        open(ARGS[1], "r") do f
            run(f)
        end
    else
        run(stdin)
    end
end

