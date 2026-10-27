using DrWatson
@quickactivate :VisualFNS

using FFTW 
using DSP
using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using StableDistributions
using SpecialFunctions
using Random
using DiffEqNoiseProcess
using LabelledArrays
using Dates
using CSV
using DataFrames
using TimeseriesMakie
using Trapz
using Base.Threads

FFTW.set_num_threads(1)  # FFTW.jl can be greedy, limit it!

const δt = 0.0001
const fs = 1.0 / δt
const xmax = 2000.
const x0 = [3.0, 0.0]
const p0 = [0.0, 0.0] 

# =====================================================================
# Main Execution / Sweep Setup
# =====================================================================

Random.seed!(42) 
num_seeds = 1000
seeds = unique(Random.rand(1:100_000_000_000, num_seeds))
# Make sure seed 27 is included to make a special trial to inspect
if !(27 in seeds)
    seeds[1] = 27 
end

Random.seed!(2404413)
shuffle!(seeds)

timespan = 100.
α_value = 1.2
β_value = 1000.0
γ_value = 1000.
τ_value = 0.002
sd = 80.0
A = Inf
β_value = 1000.0
approx_n_modes = 256


# =====================================================================
# Execute compute threads
# =====================================================================
Threads.@threads for seed in seeds
    # Run the math solver
    AdaptiveSolverSave(α_value, β_value, γ_value, timespan, seed, sd, A, τ_value, approx_n_modes; 
    well_size=3, n_proportion=0.02, target_radius=50.0, log_noise=0.0,
    x0=x0, p0=p0, xmax=xmax, δt=δt)
    
end

println("All data saved successfully.")
