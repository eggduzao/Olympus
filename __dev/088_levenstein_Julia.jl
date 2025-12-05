# Levenshtein edit distance in Julia
# Reads two lines (strings) from stdin and prints the distance

function levenshtein(s1::AbstractString, s2::AbstractString)
    m, n = length(s1), length(s2)
    dp = Array{Int}(undef, m+1, n+1)

    for i in 0:m
        dp[i+1, 1] = i
    end
    for j in 0:n
        dp[1, j+1] = j
    end

    for i in 1:m
        for j in 1:n
            cost = (s1[i] == s2[j]) ? 0 : 1
            dp[i+1, j+1] = minimum((
                dp[i, j+1] + 1,     # deletion
                dp[i+1, j] + 1,     # insertion
                dp[i, j] + cost     # substitution
            ))
        end
    end

    return dp[m+1, n+1]
end

# Read input
s1 = readline(stdin)
s2 = readline(stdin)

println(levenshtein(s1, s2))

