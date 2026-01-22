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
