"""
Simulation functions for generating synthetic matching data
"""

using ForwardDiff
using NonLinearProg
using DataFrames
using CSV
using StatsBase
using Distributions
using Random

"""
    eqconstraints(uv, mdl, σ; margins = true)

Compute equilibrium constraints for the separable matching model.

The constraints ensure that matching counts satisfy supply-demand balance:
- Sum of matches for each worker type equals total workers of that type
- Sum of matches for each firm type equals total firms of that type

# Arguments
- `uv::Vector{Float64}`: Vector of utilities `[u₁, ..., uₙₑₓ, v₁, ..., vₙₑᵧ]`
  where `u` are worker utilities and `v` are firm utilities
- `mdl::model`: Model structure containing parameters and type counts
- `σ::Vector{Float64}`: Vector `[σ₁, σ₂]` of matching elasticity parameters
- `margins::Bool`: If `true`, include type counts in matching probabilities (default: `true`)

# Returns
- `constraints::Vector{Float64}`: Vector of constraint violations `[margin_x; margin_y]`
  where `margin_x` and `margin_y` are the differences between supply and demand
  for each worker and firm type. At equilibrium, all elements should be zero.
"""
function eqconstraints(uv, mdl, σ; margins = true)

    u = uv[1:mdl.nbx]
    v = uv[mdl.nbx+1:mdl.nbz]

    sum_σ = σ[1]+σ[2]

    if margins
        μ_xy = [exp((mdl.Φ[x,y]-u[x]-v[y]+σ[1]*log(mdl.n[x])+σ[2]*log(mdl.m[y]))/sum_σ) for x=1:mdl.nbx, y=1:mdl.nby]
        μ_x0 = [exp(-u[x]/σ[1]+log(mdl.n[x])) for x=1:mdl.nbx]
        μ_0y = [exp(-v[y]/σ[2]+log(mdl.m[y])) for y=1:mdl.nby]
    else
        μ_xy = [exp((mdl.Φ[x,y]-u[x]-v[y])/sum_σ) for x=1:mdl.nbx, y=1:mdl.nby]
        μ_x0 = [exp(-u[x]/σ[1]) for x=1:mdl.nbx]
        μ_0y = [exp(-v[y]/σ[2]) for y=1:mdl.nby]
    end

    margin_x = [sum(μ_xy[x,y] for y=1:mdl.nby)+μ_x0[x] - mdl.n[x] for x=1:mdl.nbx]
    margin_y = [sum(μ_xy[x,y] for x=1:mdl.nbx)+μ_0y[y] - mdl.m[y] for y=1:mdl.nby]

    return [margin_x; margin_y]
end

"""
    equilibrium(uv, mdl, σ; margins = true)

Compute equilibrium matching counts and wages for given utilities.

# Arguments
- `uv::Vector{Float64}`: Vector of utilities `[u₁, ..., uₙₑₓ, v₁, ..., vₙₑᵧ]`
  where `u` are worker utilities and `v` are firm utilities
- `mdl::model`: Model structure containing parameters and type counts
- `σ::Vector{Float64}`: Vector `[σ₁, σ₂]` of matching elasticity parameters
- `margins::Bool`: If `true`, include type counts in matching probabilities (default: `true`)

# Returns
- `μ_xy::Matrix{Float64}`: Matrix of size `(nbx, nby)` containing matched pair counts
- `μ_x0::Vector{Float64}`: Vector of length `nbx` containing unmatched worker counts
- `μ_0y::Vector{Float64}`: Vector of length `nby` containing unmatched firm counts
- `w_xy::Matrix{Float64}`: Matrix of size `(nbx, nby)` containing equilibrium wages
  for each matched pair
"""
function equilibrium(uv, mdl, σ; margins = true)
    u = uv[1:mdl.nbx]
    v = uv[mdl.nbx+1:mdl.nbz]

    sum_σ = σ[1]+σ[2]

    if margins
        μ_xy = [exp((mdl.Φ[x,y]-u[x]-v[y]+σ[1]*log(mdl.n[x])+σ[2]*log(mdl.m[y]))/sum_σ) for x=1:mdl.nbx, y=1:mdl.nby]
        μ_x0 = [exp(-u[x]/σ[1]+log(mdl.n[x])) for x=1:mdl.nbx]
        μ_0y = [exp(-v[y]/σ[2]+log(mdl.m[y])) for y=1:mdl.nby]

        w_xy = [(σ[1]/sum_σ) .*(mdl.γ[x,y]-v[y]+σ[2]*log(mdl.m[y])) + (σ[2]/sum_σ)*(u[x]-mdl.α[x,y]-σ[1]*log(mdl.n[x])) for x=1:mdl.nbx, y=1:mdl.nby]
    else
        μ_xy = [exp((mdl.Φ[x,y]-u[x]-v[y])/sum_σ) for x=1:mdl.nbx, y=1:mdl.nby]
        μ_x0 = [exp(-u[x]/σ[1]) for x=1:mdl.nbx]
        μ_0y = [exp(-v[y]/σ[2]) for y=1:mdl.nby]

        w_xy = [(σ[1]/sum_σ) .*(mdl.γ[x,y]-v[y]) + (σ[2]/sum_σ)*(u[x]-mdl.α[x,y]) for x=1:mdl.nbx, y=1:mdl.nby]
    end

    return μ_xy, μ_x0, μ_0y, w_xy
end

"""
    social_planner_opt(mdl, σ; margins = true)

Solve for equilibrium utilities using constrained optimization.

Finds the utility vector that satisfies the equilibrium constraints (supply-demand balance)
by solving a constrained optimization problem.

# Arguments
- `mdl::model`: Model structure containing parameters and type counts
- `σ::Vector{Float64}`: Vector `[σ₁, σ₂]` of matching elasticity parameters
- `margins::Bool`: If `true`, include type counts in matching probabilities (default: `true`)

# Returns
- `uv_opt::Vector{Float64}`: Optimal utility vector `[u₁, ..., uₙₑₓ, v₁, ..., vₙₑᵧ]`
  that satisfies equilibrium constraints
- `objopt::Float64`: Objective function value at optimum (should be 0.0)
- `termstat`: Termination status from the optimization solver
  (e.g., `MathOptInterface.LOCALLY_SOLVED` if successful)

# Notes
- Uses `NonLinearProg.fmincon` with equality constraints
- Constraints are computed using `eqconstraints` function
- Jacobian is computed using automatic differentiation (`ForwardDiff`)
"""
function social_planner_opt(mdl, σ; margins = true)
    fun = function(x)
        return 0.0
    end
    gfun = function(x)
        return zeros(length(x))
    end
    
    function con(x)
        ceq = eqconstraints(x, mdl, σ; margins=margins)
        return ceq
    end
    function Jcon(x)
        ∂_con = ForwardDiff.jacobian(x) do x
            eqconstraints(x, mdl, σ; margins=margins)
        end
        return ∂_con
    end
    
    uv0 = rand(mdl.nbz)
    cons_ub = zeros(mdl.nbz)
    uv_opt, objopt, termstat = NonLinearProg.fmincon(fun, uv0; g=gfun, h=con, J=Jcon, nlcon_ub=cons_ub, nlcon_lb = cons_ub, tol=1e-12, max_iter=3000, print_summary=false, print_level=0)

    return uv_opt, objopt, termstat
end

"""
    create_dataframe(mdl, σ, N; addon="")

Generate synthetic matching data from model parameters.

Simulates a matching market by:
1. Solving for equilibrium utilities
2. Computing equilibrium matching counts and wages
3. Sampling N observations from the equilibrium distribution
4. Adding wage noise

# Arguments
- `mdl::model`: Model structure containing parameters and type counts
- `σ::Vector{Float64}`: Vector `[σ₁, σ₂]` of matching elasticity parameters
- `N::Int`: Number of observations to generate
- `addon::String`: Output directory path for CSV files. If empty (default: `""`),
  files are saved in the current directory. If specified, the directory will be
  created if it doesn't exist.

# Returns
- `df::DataFrame`: Individual-level dataframe with columns:
  - `:type_x`, `:type_y`: Worker and firm types
  - `:x_val1`, `:x_val2`, `:y_val1`, `:y_val2`: Characteristics
  - `:BF1`, `:BF2`: Basis functions (absolute differences)
  - `:mu_obs`: Matching indicator (always 1 for matched pairs)
  - `:wage_obs`: Observed wages (equilibrium wages + noise)
- `termstat`: Termination status from equilibrium optimization
  (e.g., `MathOptInterface.LOCALLY_SOLVED` if successful)

# Output Files
Two CSV files are written:
- `simu_sigma1_{σ₁}_sigma2_{σ₂}_size{N}.csv`: Individual-level data
- `simu_agg_sigma1_{σ₁}_sigma2_{σ₂}_size{N}.csv`: Aggregated matching counts

Files are saved to the directory specified by `addon` (or current directory if empty).

# Notes
- Wage noise is drawn from a standard normal distribution
- Matching probabilities are computed from equilibrium utilities
- Observations are sampled with replacement using weighted sampling
"""
function create_dataframe(mdl, σ, N; addon="")

    uv_opt, objopt, termstat = social_planner_opt(mdl, σ)
    μ_xy, μ_x0, μ_0y, w_xy = equilibrium(uv_opt, mdl, σ)

    type_x = vcat(kron(collect(1:mdl.nbx), ones(mdl.nby)), collect(1:mdl.nbx), zeros(mdl.nby))
    type_y = vcat(kron(ones(mdl.nbx), collect(1:mdl.nby)), zeros(mdl.nbx), collect(1:mdl.nby))

    μ = vcat(vec(transpose(μ_xy)), μ_x0, μ_0y)
    w = vcat(vec(transpose(w_xy)), fill(missing, mdl.nbz))

    x_val       =  Matrix{Union{Missing, Float64}}(undef, mdl.nba, mdl.nbk)
    y_val       =  Matrix{Union{Missing, Float64}}(undef, mdl.nba, mdl.nbk)

    [x_val[:,k] = vcat(kron(mdl.Xval[:,k], ones(mdl.nby)), mdl.Xval[:,k], fill(missing, mdl.nby)) for k=1:mdl.nbk]
    [y_val[:,k] = vcat(kron(ones(mdl.nbx), mdl.Yval[:,k]), fill(missing, mdl.nbx), mdl.Yval[:,k]) for k=1:mdl.nbk]

    matches     = wsample(collect(1:mdl.nba), μ, N; replace=true)

    df          = DataFrame()

    df.type_x   = type_x[matches]
    df.type_y   = type_y[matches]
    df.x_val1   = x_val[matches,1]
    df.x_val2   = x_val[matches,2]
    df.y_val1   = y_val[matches,1]
    df.y_val2   = y_val[matches,2]

    df.BF1 = abs.(df.x_val1-df.y_val1)
    df.BF2 = abs.(df.x_val2-df.y_val2)

    df.mu_obs       = ones(N)
    w_shocks        = rand(Normal(),N)
    df.wage_obs     = w[matches]+w_shocks
    # df.arc_nb       = matches

    df_agg = combine(groupby(df, [:type_x, :type_y, :BF1, :BF2]), :mu_obs => sum => :mu_obs, :wage_obs => mean => :mean_wage_obs)
    df_agg.match_type .= 0
    df_agg[(df_agg.type_y .== 0.), :match_type] .= 1
    df_agg[(df_agg.type_x .== 0.), :match_type] .= 2
    sort!(df_agg, [:match_type, :type_x, :type_y])

    # Determine output directory: use addon as path, or current directory if empty
    output_dir = addon == "" ? "." : addon
    
    # Create directory if it doesn't exist (only if a path was specified)
    if addon != "" && !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Generate filenames
    filename_base = "simu_sigma1_" * string(σ[1]) * "_sigma2_" * string(σ[2]) * "_size" * string(N)
    filename_agg = "simu_agg_sigma1_" * string(σ[1]) * "_sigma2_" * string(σ[2]) * "_size" * string(N)
    
    # Build full file paths
    filepath = joinpath(output_dir, filename_base * ".csv")
    filepath_agg = joinpath(output_dir, filename_agg * ".csv")
    
    # Write CSV files
    CSV.write(filepath, df)
    CSV.write(filepath_agg, df_agg)

    return df, termstat

end
