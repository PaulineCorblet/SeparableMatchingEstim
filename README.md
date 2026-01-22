# SeparableMatchingEstim

A Julia package for separable matching estimation.

Companion repo to "Modeling Monopsony on the Labor Market with Separable Matching Models", joint with Arnaud Dupuy.

## Overview

This package implements estimation methods for separable matching models, which are used to analyze matching markets (e.g., labor markets) where agents on both sides of the market have heterogeneous characteristics. The main functionality includes:

- **Estimation**: Estimates model parameters (σ₁, σ₂, α, γ coefficients) using a two-step procedure combining Poisson regression for matching counts and OLS for wages
- **Data preparation**: Prepares matching and wage dataframes from observed data, handling both matched and unmatched agents
- **Covariance computation**: Computes variance-covariance matrices for the estimated parameters
- **Simulations**: Includes simulation functions (`sim/functions.jl`) to generate synthetic matching data from model parameters, useful for testing and Monte Carlo studies

## Installation

**Important:** This package requires `NonLinearProg.jl`, which must be installed first as it's not in the Julia package registry:

```julia
using Pkg
Pkg.add(url="https://github.com/PaulineCorblet/NonLinearProg.git")
```

Then install `SeparableMatchingEstim` from GitHub:

```julia
Pkg.add(url="https://github.com/PaulineCorblet/SeparableMatchingEstim.git")
```

Or install from a local path:

```julia
Pkg.add(path="path/to/SeparableMatchingEstim")
```

## Usage

```julia
using SeparableMatchingEstim
```

## Dependencies

This package requires:
- **NonLinearProg.jl** - for nonlinear optimization
- **Ipopt** - nonlinear optimization solver (installed automatically via NonLinearProg.jl)
- **DataFrames.jl** - for data manipulation
- **GLM.jl** - for statistical modeling
- **CSV.jl** - for reading/writing CSV files
- **StatsBase.jl** - for statistical functions
- **Distributions.jl** - for probability distributions
- **ForwardDiff.jl** - for automatic differentiation

Ipopt is distributed under the Eclipse Public License (EPL).