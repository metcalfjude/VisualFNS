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

# Define parameter sweep values
alpha_values = 1.1:0.05:2.0
γ_values = vcat(600:50:950, 1050:50:1200) # Ignoring γ=1000, since we have this data already

# Build a flat array of all (alpha, beta, seed) combinations for threaded execution
all_tasks = collect(Iterators.product(alpha_values, γ_values, seeds))

# Separate the tasks where seed == 27
tasks_27 = filter(t -> t[3] == 27, all_tasks)
tasks_rest = filter(t -> t[3] != 27, all_tasks)

# Shuffle both lists independently for random parameter execution
Random.seed!(2404413)
shuffle!(tasks_27)
shuffle!(tasks_rest)

# Concatenate so seed 27 tasks are entirely at the front
tasks = vcat(tasks_27, tasks_rest)

println("=> Scheduled $(length(tasks_27)) visual inspection tasks (Seed 27) first.")
println("Starting Parameter Sweep... (Total Tasks: $(length(tasks)))")

timespan = 100.
β_value = 1000.
sd = 80.0
A = Inf
τ_val = 2e-4
approx_n_modes = 256

csv_path = "info_sweep-transition"
movie_path = "info_sweep-transition"

# =====================================================================
# 1. Setup the Channel and Robust Writer Task
# =====================================================================
results_channel = Channel{Any}(Inf) # Unbounded channel so compute threads never block

# Spawn on the interactive thread pool, and use a try-catch to prevent silent failures
writer_task = Threads.@spawn :interactive begin
    master_freqs_saved = false
    
    # This loop automatically terminates when close(results_channel) is called
    for res in results_channel
        # Write immediately - no batching
        CSV.write(res.summary_path, res.summary_df; append=isfile(res.summary_path))
        
        if !isnothing(res.power_df)
            CSV.write(res.power_path, res.power_df; append=isfile(res.power_path))
            
            if !master_freqs_saved && !isnothing(res.freqs)
                freq_path = datadir(csv_path, "master_frequencies.csv")
                if !isfile(freq_path)
                    CSV.write(freq_path, DataFrame(frequency = res.freqs))
                end
                master_freqs_saved = true
            end
        end
    end
end

println("Starting Robust Asynchronous Parameter Sweep...")

# =====================================================================
# 2. Compute Loop with Graceful Shutdown
# =====================================================================
try
    Threads.@threads for (α_val, γ_val, seed) in tasks
        # Run the math solver
        res = AdaptiveSolver(α_val, β_value, γ_val, timespan, seed, sd, A, τ_val, approx_n_modes; 
            well_size=3, n_proportion=0.02, target_radius=50.0, log_noise=0.0,
            x0=x0, p0=p0, xmax=xmax, δt=δt,
            csv_folder = csv_path, movie_folder = movie_path)
        
        # Instantly hand off to the writer task
        put!(results_channel, res)
    end
catch e
    if e isa InterruptException
        println("\n=> Sweep aborted by user (^C)! Initiating safe shutdown...")
    else
        println("\n=> Fatal error in compute loop!")
        rethrow(e)
    end
finally
    # This block executes NO MATTER HOW the loop stops.
    # It closes the channel, which tells the writer_task to finish its queue and exit.
    println("=> Closing results channel. Waiting for background writer to flush remaining data...")
    close(results_channel) 
    wait(writer_task) 
    println("=> Safe shutdown complete. All data successfully written to disk.")
end
