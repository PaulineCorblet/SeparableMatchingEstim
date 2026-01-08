"""
Model structures and helper functions
"""

"""
    model

Structure to hold model parameters and data.
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

Create an observation model from matching dataframes.
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

