"""
Model structures and helper functions
"""

"""
    model

Structure to hold model parameters and data for separable matching models.

# Fields
- `nbx::Int`: Number of worker types (x types)
- `nby::Int`: Number of firm types (y types)
- `nbz::Int`: Total number of agent types (nbx + nby)
- `nba::Int`: Total number of arcs (nbx*nby + nbx + nby)
- `nbk::Int`: Number of basis functions/dimensions
- `n::Vector`: Vector of length `nbx` containing counts of each worker type
- `m::Vector`: Vector of length `nby` containing counts of each firm type
- `Xval::Matrix`: Matrix of size `(nbx, nbk)` containing worker characteristics
- `Yval::Matrix`: Matrix of size `(nby, nbk)` containing firm characteristics
- `κ::Vector`: Vector of length `nbk` containing worker preference parameters
- `λ::Vector`: Vector of length `nbk` containing firm preference parameters
- `α::Matrix`: Matrix of size `(nbx, nby)` containing worker matching utilities
- `γ::Matrix`: Matrix of size `(nbx, nby)` containing firm matching utilities
- `Φ::Matrix`: Matrix of size `(nbx, nby)` containing total matching surplus (α + γ)
"""
struct model
    nbx
    nby
    nbz
    nba
    nbk
    n
    m
    Xval
    Yval
    κ
    λ
    α
    γ
    Φ
end

"""
    obs_model(df_match_xy, df_match_x0, df_match_0y)

Create a `model` structure from observed matching dataframes.

# Arguments
- `df_match_xy::DataFrame`: DataFrame containing matched pairs (type_x > 0, type_y > 0)
  with columns: `:type_x`, `:type_y`, `:x_val1`, `:x_val2`, `:y_val1`, `:y_val2`, `:count`
- `df_match_x0::DataFrame`: DataFrame containing unmatched workers (type_y == 0)
  with columns: `:type_x`, `:type_y`, `:x_val1`, `:x_val2`, `:count`
- `df_match_0y::DataFrame`: DataFrame containing unmatched firms (type_x == 0)
  with columns: `:type_x`, `:type_y`, `:y_val1`, `:y_val2`, `:count`

# Returns
- `obs_mdl::model`: A `model` structure initialized with:
  - Dimensions inferred from the dataframes
  - Worker and firm type counts (`n` and `m`) computed from matching counts
  - Characteristic matrices (`Xval`, `Yval`) extracted from the dataframes
  - Preference parameters (`κ`, `λ`, `α`, `γ`, `Φ`) initialized to zero (to be estimated)
"""
function obs_model(df_match_xy, df_match_x0, df_match_0y)
    nbx = size(df_match_x0, 1)
    nby = size(df_match_0y, 1)
    nbz = nbx + nby
    nba = nbx * nby + nbx + nby
    nbk = 2

    n = combine(groupby(df_match_xy, :type_x), :count => sum).count_sum + df_match_x0.count
    m = combine(groupby(df_match_xy, :type_y), :count => sum).count_sum + df_match_0y.count

    Xval = Matrix(df_match_xy[df_match_xy.type_y .== 1, [:x_val1, :x_val2]])
    Yval = Matrix(df_match_xy[df_match_xy.type_x .== 1, [:y_val1, :y_val2]])

    obs_mdl = model(nbx, nby, nbz, nba, nbk, n, m, Xval, Yval, 0., 0., 0., 0., 0.)
    return obs_mdl
end

