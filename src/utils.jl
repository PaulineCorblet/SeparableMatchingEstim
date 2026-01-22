"""
Helper utilities for SeparableMatchingEstim
"""

"""
    indmat(x)

Create an indicator matrix from a vector of categorical values.

# Arguments
- `x::Vector`: A vector of categorical values (typically integers representing types)

# Returns
- `M::Matrix{Int}`: An indicator matrix where:
  - Each row corresponds to an observation in `x`
  - Each column corresponds to a category (1, 2, ..., max(x))
  - `M[i, j] = 1` if `x[i] == j`, otherwise `M[i, j] = 0`
  - Zero values in `x` result in all zeros in the corresponding row

# Examples
```julia
x = [1, 2, 1, 3, 2]
M = indmat(x)
# Returns a 5×3 matrix with ones indicating category membership
```
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

