"""
    SeparableMatchingEstim

A Julia package for separable matching estimation.
"""
module SeparableMatchingEstim

using DataFrames
using StatsModels
using LinearAlgebra
using GLM
using CovarianceMatrices

# Include source files
include("utils.jl")
include("models.jl")
include("data_prep.jl")
include("estimation.jl")

# Export main functions
export run_estimation
export compute_vcov
export create_matching_wage_df_CD
export obs_model
export model
export indmat

end # module

