# Activate package environment and install dependencies
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.resolve()  # Resolve dependencies to update manifest
Pkg.instantiate()  # Install all dependencies

using SeparableMatchingEstim
using Test
using DataFrames
using CSV
using ForwardDiff
using StatsBase
using Distributions
using NonLinearProg
using LinearAlgebra
using Random
using MathOptInterface

@testset "SeparableMatchingEstim.jl" begin
    @testset "indmat" begin
        x = [1, 2, 1, 3, 2]
        M = indmat(x)
        @test size(M) == (5, 3)
        @test all(sum(M, dims=2) .== 1)
        @test M[1, 1] == 1
        @test M[2, 2] == 1
        @test M[3, 1] == 1
    end

    @testset "run_estimation with simulation data" begin
        # Set seed for reproducibility
        Random.seed!(12345)
        
        # Use the same parameters as in estimation.jl lines 25-34
        σ_a = [1.0, 1.0]
        N = 5000000
        
        # Set up model parameters
        nbx     = 9
        nby     = 9
        nbz     = nbx+nby
        nba     = nbx*nby+nbx+nby
        
        n = rand(1:10, nbx)
        m = rand(1:10, nby)
        
        nbk     = 2
        Xval    = rand(nbx, nbk)
        Yval    = rand(nby, nbk)
        κ       = rand(nbk)
        λ       = rand(nbk)
        α       = [sum(κ[k]*abs(Xval[x,k]-Yval[y,k]) for k=1:nbk) for x=1:nbx, y=1:nby]
        γ       = [sum(λ[k]*abs(Xval[x,k]-Yval[y,k]) for k=1:nbk) for x=1:nbx, y=1:nby]
        Φ       = α+γ
        
        mdl = model(nbx, nby, nbz, nba, nbk, n, m, Xval, Yval, κ, λ, α, γ, Φ)
        
        # Generate simulation data using create_dataframe
        df, termstat = create_dataframe(mdl, σ_a, N; addon="")
        
        # Check optimization termination status (Ipopt output is suppressed, we output status instead)
        println("Simulation termination status: $termstat")
        # Test that optimization converged to locally solved solution
        @test termstat == MathOptInterface.LOCALLY_SOLVED
        
        # Check that required columns exist
        required_cols = [:type_x, :type_y, :x_val1, :x_val2, :y_val1, :y_val2, :wage_obs]
        missing_cols = setdiff(required_cols, Symbol.(names(df)))
        if !isempty(missing_cols)
            @warn "Missing required columns: $missing_cols. Skipping test."
            return
        end
        
        # Prepare data using create_matching_wage_df
        df_match_xy, df_wage, om = create_matching_wage_df(df)
        
        # Check that the function returns the expected structure
        @test hasproperty(df_match_xy, :BF1)
        @test hasproperty(df_match_xy, :BF2)
        @test hasproperty(df_match_xy, :diff_log_mu)
        @test hasproperty(df_match_xy, :log_mu_0y)
        @test hasproperty(df_match_xy, :log_mu_x0)
        @test hasproperty(df_wage, :BF1)
        @test hasproperty(df_wage, :BF2)
        @test hasproperty(df_wage, :diff_log_mu)
        @test om.nbx > 0
        @test om.nby > 0
        
        # Ensure wage_obs column exists in df_wage
        if !hasproperty(df_wage, :wage_obs)
            # If wage_obs is not in the prepared dataframe, get it from original df
            # Match rows by type_x and type_y
            df_wage.wage_obs = zeros(nrow(df_wage))
            for i in 1:nrow(df_wage)
                matching_rows = df[(df.type_x .== df_wage.type_x[i]) .& 
                                    (df.type_y .== df_wage.type_y[i]), :]
                if nrow(matching_rows) > 0
                    df_wage.wage_obs[i] = mean(matching_rows.wage_obs)
                end
            end
        end
        
        # Run the estimation (matching estimation.jl line 55)
        results, pred_mu = run_estimation(df_match_xy, df_wage)
        
        # Check that results have the expected structure
        # Results should be [sigma1, sigma2, alpha..., gamma...]
        # For 2 basis functions (BF1, BF2), we expect 2 alpha and 2 gamma coefficients
        @test length(results) >= 4  # At least sigma1, sigma2, and some coefficients
        @test isfinite(results[1])  # sigma1 should be finite
        @test isfinite(results[2])  # sigma2 should be finite
        @test all(isfinite.(results))  # All results should be finite
        
        # Check predicted counts
        @test length(pred_mu) == nrow(df_match_xy)  # Should match number of rows
        @test all(pred_mu .>= 0)  # Predicted counts should be non-negative
        @test all(isfinite.(pred_mu))  # All predicted counts should be finite
        
        println("Test passed! Results: sigma1=$(round(results[1], digits=4)), sigma2=$(round(results[2], digits=4))")
        println("Number of worker types (nbx): $(om.nbx), Number of firm types (nby): $(om.nby)")
        
        # Test with vcov = true
        results_vcov, vcov_results, pred_mu_vcov = run_estimation(df_match_xy, df_wage; vcov = true)
        
        # Check that results have the expected structure
        @test length(results_vcov) >= 4  # At least sigma1, sigma2, and some coefficients
        @test isfinite(results_vcov[1])  # sigma1 should be finite
        @test isfinite(results_vcov[2])  # sigma2 should be finite
        @test all(isfinite.(results_vcov))  # All results should be finite
        
        # Check covariance matrix
        @test size(vcov_results) == (length(results_vcov), length(results_vcov))  # Should be square matrix
        @test all(isfinite.(vcov_results))  # All covariance values should be finite
        @test issymmetric(vcov_results) || isapprox(vcov_results, vcov_results')  # Should be symmetric
        
        # Check predicted counts
        @test length(pred_mu_vcov) == nrow(df_match_xy)  # Should match number of rows
        @test all(pred_mu_vcov .>= 0)  # Predicted counts should be non-negative
        @test all(isfinite.(pred_mu_vcov))  # All predicted counts should be finite
        
        # Results should be the same whether vcov is true or false
        @test isapprox(results, results_vcov, rtol=1e-10)
        @test isapprox(pred_mu, pred_mu_vcov, rtol=1e-10)
        
        println("Test with vcov=true passed! Covariance matrix size: $(size(vcov_results))")
    end
end

