"""
Estimation functions for separable matching models
"""

"""
    run_estimation(df_match_xy, df_wage, om)

Run the estimation procedure with all data (matched and unmatched).
Returns a tuple with:
- results: a vector with [sigma1, sigma2, alpha..., gamma...]
- pred_count: predicted counts from the Poisson regression
"""
function run_estimation(df_match_xy, df_wage; vcov = false)

    bf_cols = [col for col in names(df_match_xy) if startswith(string(col), "BF")]
    bf_cols_sorted = sort(bf_cols, by = x -> parse(Int, replace(string(x), "BF" => "")))
    if isempty(bf_cols_sorted)
        error("No basis function columns (BF1, BF2, ...) found in df_match_xy")
    end
    nb_bf = length(bf_cols)
    
    poisson = glm(@formula(count ~ 0 + BF1 + BF2 + diff_log_mu), 
                  df_match_xy, Poisson(), 
                  offset = df_match_xy[:, :log_mu_0y], 
                  minstepfac = 0.0001)

    ols = lm(@formula(wage_obs ~ 0 + BF1 + BF2 + diff_log_mu), df_wage)

    coef_poisson = coef(poisson)
    coef_ols = coef(ols) 
    hat_sigma1 = -coef_ols[end] / (1 - coef_poisson[end])
    hat_sigma2 = -coef_ols[end] / coef_poisson[end]

    hat_alpha = hat_sigma1 * coef_poisson[1:end-1] - coef_ols[1:end-1]
    hat_gamma = hat_sigma2 * coef_poisson[1:end-1] + coef_ols[1:end-1]

    results = vcat(hat_sigma1, hat_sigma2, hat_alpha, hat_gamma)
    
    # Always compute pred_mu for return value
    pred_mu = predict(poisson)

    if vcov

        pred_w = predict(ols)

        Z_xy = hcat(Matrix(df_match_xy[:, bf_cols_sorted]), df_match_xy.diff_log_mu)
        W_ij = hcat(Matrix(df_wage[:, bf_cols_sorted]), df_wage.diff_log_mu)

        hat_A_poisson = ((pred_mu .* Z_xy)' * Z_xy)
        hat_A_ols = (W_ij' * W_ij)

        # hat_A = vcat(hcat(hat_A_poisson, zeros(nb_bf+1, nb_bf+1)), hcat(zeros(nb_bf+1, nb_bf+1), hat_A_ols))
        inv_hat_A_poisson = inv(hat_A_poisson)
        inv_hat_A_ols = inv(hat_A_ols)
        inv_hat_A = vcat(hcat(inv_hat_A_poisson, zeros(nb_bf+1, nb_bf+1)), hcat(zeros(nb_bf+1, nb_bf+1), inv_hat_A_ols))

        s_xy = (df_match_xy.count - pred_mu) .* Z_xy
        m_ij = (df_wage.wage_obs - pred_w) .* W_ij

        M = indmat(df_wage.match_type)
        m_xy = M'*m_ij

        S_poisson = ((s_xy)' * s_xy)
        S_ols = ((m_ij)' * m_ij)
        S_cross = (s_xy' * m_xy)
        
        S = vcat(hcat(S_poisson, S_cross), hcat(S_cross', S_ols))

        pre_vcov = inv_hat_A * S * inv_hat_A'
        pre_vcov_sym = (pre_vcov + pre_vcov') / 2
        E, V = eigen(pre_vcov_sym)
        E_clipped = max.(E, 0.0)
        pre_vcov_psd = V * Diagonal(E_clipped) * V'
        
        δ_p = coef_poisson[end]
        δ_o = coef_ols[end]
        β_p = coef_poisson[1:end-1]
        # β_o = coef_ols[1:end-1]
        
        n_params = 2*nb_bf + 2
        J = zeros(n_params, n_params)
        
        J[1, nb_bf+1] = -δ_o / (1 - δ_p)^2
        J[1, 2*nb_bf+2] = -1 / (1 - δ_p)
        
        J[2, nb_bf+1] = δ_o / δ_p^2
        J[2, 2*nb_bf+2] = -1 / δ_p
        
        for i in 1:nb_bf
            J[2 + i, i] = hat_sigma1
            J[2 + i, nb_bf+1] = -β_p[i] * δ_o / (1 - δ_p)^2
            J[2 + i, nb_bf+1+i] = -1
            J[2 + i, 2*nb_bf+2] = β_p[i] * (-1 / (1 - δ_p))
        end
        
        for i in 1:nb_bf
            J[2 + nb_bf + i, i] = hat_sigma2
            J[2 + nb_bf + i, nb_bf+1] = β_p[i] * δ_o / δ_p^2
            J[2 + nb_bf + i, nb_bf+1+i] = 1
            J[2 + nb_bf + i, 2*nb_bf+2] = β_p[i] * (-1 / δ_p)
        end
        
        vcov_results = J * pre_vcov_psd * J'

        return results, vcov_results, pred_mu
    else
        return results, pred_mu
    end
    
end

function compute_vcov(df_match_xy, df_wage, pred_mu)

    nbXY = size(df_match_xy)[1]

    bf_cols = [col for col in names(df_match_xy) if startswith(string(col), "BF")]
    bf_cols_sorted = sort(bf_cols, by = x -> parse(Int, replace(string(x), "BF" => "")))
    if isempty(bf_cols_sorted)
        error("No basis function columns (BF1, BF2, ...) found in df_match_xy")
    end
    Z_xy = hcat(Matrix(df_match_xy[:, bf_cols_sorted]), df_match_xy.diff_log_mu)

    hat_H = -((pred_mu .* Z_xy)' * Z_xy)/nbXY
    s_xy = (df_match_xy.count - pred_mu) .* Z_xy
    hat_S = ((s_xy)' * s_xy)/nbXY
    
    # Compute covariance matrix: inv(hat_H) * hat_S * inv(hat_H)
    inv_H = inv(hat_H)
    vcov_poisson = inv_H * hat_S * inv_H

    nbw = size(df_wage)[1]
    hat_A = ()/nbw
    
end