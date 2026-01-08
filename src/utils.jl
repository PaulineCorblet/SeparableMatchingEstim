"""
Helper utilities for SeparableMatchingEstim
"""

"""
    indmat(x)

Create an indicator matrix from a vector of categorical values.
Each row corresponds to an observation, each column to a category.
The columns are ordered 1, 2, ..., max(x) to match type indexing.
"""
function indmat(x)
    n = length(x)
    max_val = Int(maximum(x))
    M = zeros(Int, n, max_val)
    for (i, val) in enumerate(x)
        if val > 0  # Only set indicator for non-zero types
            M[i, Int(val)] = 1
        end
    end
    return M
end

