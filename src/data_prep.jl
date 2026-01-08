"""
Data preparation functions for matching estimation
"""

"""
    create_matching_wage_df_CD(df)

Create matching and wage dataframes for Choo-Siow estimation.
Returns df_match_xy, df_wage, and obs_model.
"""
function create_matching_wage_df_CD(df)
    
    nbx = Int64(findmax(df.type_x)[1])
    nby = Int64(findmax(df.type_y)[1])

    df_match_xy = combine(groupby(df[(df.type_x .> 0) .& (df.type_y .> 0), :], 
                                   [:type_x, :type_y, :x_val1, :x_val2, :y_val1, :y_val2]), 
                          nrow => :count)
    sort!(df_match_xy, [:type_x, :type_y])

    df_match_xy.BF1 = [abs(df_match_xy.x_val1[xy] - df_match_xy.y_val1[xy]) for xy = 1:(nbx*nby)]
    df_match_xy.BF2 = [abs(df_match_xy.x_val2[xy] - df_match_xy.y_val2[xy]) for xy = 1:(nbx*nby)]

    df_match_x0 = combine(groupby(df[(df.type_y .== 0), :], 
                                  [:type_x, :type_y, :x_val1, :x_val2, :y_val1, :y_val2]), 
                         nrow => :count)
    if size(df_match_x0, 1) < nbx
        missing_x = setdiff(collect(1:nbx), df_match_x0.type_x)
        for x in missing_x
            sub_match_x0 = select(df_match_xy[df_match_xy.type_x .== x, :], Not([:BF1, :BF2]))[1, :]
            sub_match_x0.count = 0
            sub_match_x0.y_val1 = missing
            sub_match_x0.y_val2 = missing
            sub_match_x0.type_y = 0.
            push!(df_match_x0, sub_match_x0)
        end
    end
    sort!(df_match_x0, [:type_x])
    
    df_match_0y = combine(groupby(df[(df.type_x .== 0), :], 
                                  [:type_x, :type_y, :x_val1, :x_val2, :y_val1, :y_val2]), 
                         nrow => :count)
    if size(df_match_0y, 1) < nby
        missing_y = setdiff(collect(1:nby), df_match_0y.type_y)
        for y in missing_y
            sub_match_0y = select(df_match_xy[df_match_xy.type_y .== y, :], Not([:BF1, :BF2]))[1, :]
            sub_match_0y.count = 0
            sub_match_0y.x_val1 = missing
            sub_match_0y.x_val2 = missing
            sub_match_0y.type_x = 0.
            push!(df_match_0y, sub_match_0y)
        end
    end
    sort!(df_match_0y, [:type_y])

    df_match_xy = hcat(df_match_xy, DataFrame(kron(ones(nbx, 1), log.(df_match_0y.count)), [:log_mu_0y]))
    df_match_xy = hcat(df_match_xy, DataFrame(kron(log.(df_match_x0.count), ones(nby, 1)), [:log_mu_x0]))

    df_match_xy.diff_log_mu = df_match_xy.log_mu_x0 - df_match_xy.log_mu_0y

    df_wage = df[(df.type_x .> 0) .& (df.type_y .> 0), :]
    df_wage.BF1 = abs.(df_wage.x_val1 - df_wage.y_val1)
    df_wage.BF2 = abs.(df_wage.x_val2 - df_wage.y_val2)
    
    # Add match_type column: sequential numbering of (type_x, type_y) combinations
    # match_type = (type_x - 1) * nby + type_y
    # This gives: (1,1)=1, (1,2)=2, ..., (1,nby)=nby, (2,1)=nby+1, etc.
    df_wage.match_type = (df_wage.type_x .- 1) .* nby .+ df_wage.type_y

    # Ensure df_match_x0 and df_match_0y are sorted and have the right length
    sort!(df_match_x0, :type_x)
    sort!(df_match_0y, :type_y)
    
    # Create indicator matrices and multiply
    # indmat creates (n_obs, nbx) matrix, transpose gives (nbx, n_obs)
    # log.(df_match_x0.count) is (nbx,) vector
    # Result should be (n_obs,) vector
    log_mu_x0_vec = log.(df_match_x0.count)
    log_mu_0y_vec = log.(df_match_0y.count)
    
    # Ensure vectors are column vectors for matrix multiplication
    df_wage.log_mu_x0 = vec(indmat(df_wage.type_x) * log_mu_x0_vec)
    df_wage.log_mu_0y = vec(indmat(df_wage.type_y) * log_mu_0y_vec)

    df_wage.diff_log_mu = df_wage.log_mu_x0 - df_wage.log_mu_0y

    obs_mdl = obs_model(df_match_xy, df_match_x0, df_match_0y)

    return df_match_xy, df_wage, obs_mdl
end

