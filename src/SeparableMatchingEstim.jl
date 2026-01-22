"""
    SeparableMatchingEstim

A Julia package for separable matching estimation.
"""
module SeparableMatchingEstim

using DataFrames
using StatsModels
using LinearAlgebra
using GLM

# Include source files
include("utils.jl")
include("models.jl")
include("data_prep.jl")
include("estimation.jl")
include("simulation.jl")

# Export main functions
export run_estimation
export compute_vcov
export create_matching_wage_df
export obs_model
export model
export indmat
# Simulation functions
export create_dataframe
export eqconstraints
export equilibrium
export social_planner_opt

end # module

