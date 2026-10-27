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


println("Starting Seed Sweep... (Total Tasks: $(length(seeds)))")

timespan = 100.
α_value = 1.9
γ_value = 1000.
sd = 80.0
A = Inf
β_value = 1500.0
τ_value = 1e-4
approx_n_modes = 128

csv_folder = "time-avg-auec"
movie_folder = "time-avg-auec-movies"

# =====================================================================
# Set up asynchronous CSV logging
# =====================================================================
results_channel = Channel{Any}(1000)

writer_task = Threads.@spawn begin
    summary_batches = Dict{String, DataFrame}()
    power_batches = Dict{String, DataFrame}()
    batch_count = 0
    master_freqs_saved = false
    
    for res in results_channel
        # Group summary data by dynamic file path
        if !haskey(summary_batches, res.summary_path)
            summary_batches[res.summary_path] = res.summary_df
        else
            append!(summary_batches[res.summary_path], res.summary_df)
        end
        
        # Group power data by dynamic file path
        if !isnothing(res.power_df)
            if !haskey(power_batches, res.power_path)
                power_batches[res.power_path] = res.power_df
            else
                append!(power_batches[res.power_path], res.power_df)
            end
            
            # Save frequencies once for the entire sweep
            if !master_freqs_saved && !isnothing(res.freqs)
                freq_path = datadir(csv_folder, "master_frequencies.csv")
                if !isfile(freq_path)
                    CSV.write(freq_path, DataFrame(frequency = res.freqs))
                end
                master_freqs_saved = true
            end
        end
        
        batch_count += 1
        
        # Flush all accumulated DataFrames to disk every 50 runs
        if batch_count >= 50
            for (path, df) in summary_batches
                CSV.write(path, df; append=isfile(path))
            end
            empty!(summary_batches)
            
            for (path, df) in power_batches
                CSV.write(path, df; append=isfile(path))
            end
            empty!(power_batches)
            
            batch_count = 0
            println("=> Disk writer flushed batch to disk.")
        end
    end
    
    # Final flush to capture any remaining runs after the channel closes
    for (path, df) in summary_batches
        if nrow(df) > 0
            CSV.write(path, df; append=isfile(path))
        end
    end
    for (path, df) in power_batches
        if nrow(df) > 0
            CSV.write(path, df; append=isfile(path))
        end
    end
    println("=> Final disk write complete.")
end

# =====================================================================
# Execute compute threads
# =====================================================================
Threads.@threads for seed in seeds
    # Run the math solver
    res = AdaptiveSolver(α_value, β_value, γ_value, timespan, seed, sd, A, τ_value, approx_n_modes; 
    well_size=1.5, n_proportion=0.02, target_radius=50.0, noise_level=20.0, log_noise=0.0,
    x0=x0, p0=p0, xmax=xmax, δt=δt,
    csv_folder = csv_folder, movie_folder = movie_folder,)
    
    # Instantly push the result to the channel for writing
    put!(results_channel, res)
end

println("Parameter sweep math complete! Closing channel...")
close(results_channel) 
wait(writer_task) 

println("All data saved successfully.")
