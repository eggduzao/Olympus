# binary_search_cli.jl
#
# Binary search over a sorted list of numbers (ascending).
#
# USAGE (pick one):
#   # A) From Julia with a file containing one number per line:
#   julia binary_search_cli.jl 11 data.txt
#
#   # B) From Julia REPL, passing a vector directly:
#   include("binary_search_cli.jl"); binary_search_cli(11, [1,3,4,7,9,11,15])
#
# BEHAVIOR:
#   • O(log N) classic binary search.
#   • Reports FIRST occurrence if duplicates exist.
#   • On hit:  prints  FOUND <target> at index <i> (1-based)
#   • On miss: prints  NOT FOUND <target>. Insertion index <i> (1-based), between <left> and <right>
#
# NOTES:
#   • Input must be non-decreasing (ascending, duplicates OK).
#   • Handles large files by streaming lines with minimal allocations.

# -----------------------------
# Public CLI-style entry point
# -----------------------------
function binary_search_cli(target::Real, src)
    vec = _materialize_numbers(src)
    _ensure_ascending!(vec)

    t = float(target)
    idx, found, ins = binary_search(vec, t; first_occurrence=true)

    if found
        @printf("FOUND %g at index %d (1-based)\n", t, idx)
    else
        left  = ins > 1           ? string(vec[ins-1]) : "-inf"
        right = ins <= length(vec) ? string(vec[ins])   : "+inf"
        @printf("NOT FOUND %g. Insertion index %d (1-based), between %s and %s\n",
                t, ins, left, right)
    end
    return nothing
end

# ---------------------------------------
# Programmatic API (returns tuple values)
# ---------------------------------------
"""
    binary_search(vec::AbstractVector{<:Real}, target::Real; first_occurrence=true)
-> (idx::Int, found::Bool, insert_pos::Int)

Binary search on ascending `vec`.

- If `first_occurrence` is true and duplicates exist, returns the first matching index.
- `insert_pos` is the 1-based position where `target` should be inserted to keep order.

Complexity: O(log N).
"""
function binary_search(vec::AbstractVector{<:Real}, target::Real; first_occurrence::Bool=true)
    lo::Int = 1
    hi::Int = length(vec)
    found_idx::Int = 0
    tgt = float(target)

    while lo <= hi
        mid = fld(lo + hi, 2)                # integer midpoint
        val = float(vec[mid])
        if val == tgt
            found_idx = mid
            if first_occurrence
                hi = mid - 1                 # keep searching left side
            else
                break
            end
        elseif val < tgt
            lo = mid + 1
        else
            hi = mid - 1
        end
    end

    found = found_idx != 0
    insert_pos = lo                           # standard "insertion point"
    return found ? found_idx : 0, found, insert_pos
end

# -----------------------------
# Helpers: I/O and validation
# -----------------------------
# Accept either a filename (String/AbstractString) or a numeric vector
function _materialize_numbers(src)
    if src isa AbstractVector{<:Real}
        return collect(float.(src))  # dense Vector{Float64}
    elseif src isa AbstractString
        return _read_numbers_from_file(String(src))
    else
        throw(ArgumentError("Second argument must be a filename (String) or a numeric vector."))
    end
end

function _read_numbers_from_file(path::String)
    isfile(path) || error("File not found: $path")
    vec = Vector{Float64}()
    vecsizehint = 0
    # Optional: try to hint capacity using file size (rough heuristic)
    try
        approx_lines = max(1, Int64(filesize(path) ÷ 8))  # crude
        vecsizehint = min(approx_lines, 10^7)
        sizehint!(vec, vecsizehint)
    catch
        # ignore if filesystem doesn't support filesize
    end

    open(path, "r") do io
        for (i, line) in enumerate(eachline(io))
            s = strip(line)
            isempty(s) && continue
            v = tryparse(Float64, s)
            if v === nothing
                @warn "Skipping non-numeric line" line_no=i content=line
                continue
            end
            push!(vec, v)
        end
    end
    return vec
end

function _ensure_ascending!(v::Vector{Float64})
    # Verify non-decreasing order without allocating `diff(v)`
    last = -Inf
    @inbounds for x in v
        if x < last
            error("Input data is not in ascending order (required for binary search).")
        end
        last = x
    end
    return nothing
end

# -----------------------------
# CLI glue for `julia script.jl`
# -----------------------------
function main()
    if length(ARGS) == 0
        println("Usage:")
        println("  julia binary_search_cli.jl <target_number> <file_with_one_number_per_line>")
        println("  julia binary_search_cli.jl <target_number>     # reads from STDIN")
        println("\nExample:")
        println("  printf \"1\\n3\\n4\\n7\\n9\\n11\\n15\\n\" | julia binary_search_cli.jl 11")
        return
    end

    target_str = ARGS[1]
    tgt = tryparse(Float64, target_str)
    tgt === nothing && error("Target must be numeric, got: $target_str")

    if length(ARGS) >= 2
        src = ARGS[2]
        return binary_search_cli(tgt, src)
    else
        # Read from STDIN if no file provided
        tmp = Vector{Float64}()
        for (i, line) in enumerate(eachline(stdin))
            s = strip(line)
            isempty(s) && continue
            v = tryparse(Float64, s)
            v === nothing && @warn "Skipping non-numeric line from STDIN" line_no=i content=line
            v !== nothing && push!(tmp, v)
        end
        return binary_search_cli(tgt, tmp)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
