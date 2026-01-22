"""
    SeparableMatchingEstim

A Julia package for separable matching estimation.
"""
module SeparableMatchingEstim

using DataFrames
using StatsModels
using LinearAlgebra
using GLM
using ForwardDiff
using NonLinearProg
using CSV
using StatsBase
using Distributions
using Random

# Include source files
include("simulation.jl")
include("utils.jl")
include("models.jl")
include("data_prep.jl")
include("estimation.jl")

# Simulation functions
export create_dataframe
export eqconstraints
export equilibrium
export social_planner_opt

# Export main functions
export run_estimation
export compute_vcov
export create_matching_wage_df
export obs_model
export model
export indmat

end # module

