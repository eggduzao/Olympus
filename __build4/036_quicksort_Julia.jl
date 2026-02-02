# quicksort_inplace.jl
# In-place quicksort on an array in Julia.
# - If called with a filename (string/Path), reads one number per line.
# - If called with no args, it reads numbers from STDIN (one per line).
# - Prints the sorted numbers, one per line, to STDOUT.
#
# Usage (file):
#   julia quicksort_inplace.jl input.txt
#
# Usage (stdin):
#   printf "5\n3\n8\n1\n2\n" | julia quicksort_inplace.jl
#
# Notes:
# - Uses Hoare partition scheme (good for many duplicates).
# - Tail-recursion elimination via loop to avoid deep recursion.
# - Skips blank lines and lines starting with '#'.

# --------------------------
# Parsing helpers
# --------------------------
"""
    read_numbers(source)::Vector{Float64}

Read one number per line from a filename (`AbstractString`) or from
an `IO` (e.g., `stdin`). Lines that are empty or start with `#` are skipped.
"""
function read_numbers(source)
    nums = Float64[]
    io = source isa AbstractString ? open(source, "r") : source
    local closer = nothing
    if source isa AbstractString
        closer = () -> close(io)
    else
        closer = () -> nothing
    end
    try
        for ln in eachline(io)
            s = strip(ln)
            isempty(s) && continue
            startswith(s, '#') && continue
            push!(nums, parse(Float64, s))
        end
    finally
        closer()
    end
    return nums
end

# --------------------------
# In-place quicksort (Hoare)
# --------------------------
@inline function swap!(A, i, j)
    A[i], A[j] = A[j], A[i]
    return nothing
end

"""
    partition_hoare!(A, lo, hi)::Int

Hoare partitioning. Returns index `p` such that `A[lo:p] <= pivot <= A[p+1:hi]`.
"""
function partition_hoare!(A::AbstractVector{T}, lo::Int, hi::Int) where {T}
    mid = (lo + hi) >>> 1  # fast div by 2
    pivot = A[mid]
    i = lo - 1
    j = hi + 1
    while true
        i += 1
        while A[i] < pivot
            i += 1
        end
        j -= 1
        while A[j] > pivot
            j -= 1
        end
        if i >= j
            return j
        end
        swap!(A, i, j)
    end
end

"""
    quicksort!(A)

Sort vector `A` in-place using quicksort with Hoare partitioning.
"""
function quicksort!(A::AbstractVector{T}) where {T}
    n = length(A)
    n ≤ 1 && return A

    # emulate recursion with an explicit stack of (lo, hi) ranges
    stack = Vector{Tuple{Int,Int}}()
    push!(stack, (1, n))

    while !isempty(stack)
        lo, hi = pop!(stack)
        while lo < hi
            p = partition_hoare!(A, lo, hi)
            # Recurse (push) on the larger side last to keep stack shallow
            left_len  = p - lo + 1
            right_len = hi - (p + 1) + 1
            if left_len < right_len
                # Sort left first, then loop on right
                if lo < p; push!(stack, (lo, p)); end
                lo = p + 1
            else
                # Sort right first, then loop on left
                if p + 1 < hi; push!(stack, (p + 1, hi)); end
                hi = p
            end
        end
    end
    return A
end

# --------------------------
# Entry point
# --------------------------
function main()
    if length(ARGS) ≥ 1
        A = read_numbers(ARGS[1])
    else
        A = read_numbers(stdin)
    end
    quicksort!(A)
    # print one per line, preserving integer-looking values neatly
    for x in A
        if isfinite(x) && x == round(x)
            println(Int(round(x)))
        else
            println(x)
        end
    end
end

# run only when executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

