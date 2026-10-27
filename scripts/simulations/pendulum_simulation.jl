using DrWatson
@quickactivate :VisualFNS

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
using Dates
using CSV
using DataFrames
using TimeseriesMakie

const α_value = 1.3
const β_value = 0.8
const γ_value = 1.4
const TIMESPAN = 10000.0

const RADIUS = 40
const x0 = [3.0, 0.0] 
const p0 = [0.0, 0.0] 
const ω = 0.4			# The angular frequency of the circular motion, previously called 'k'
const SIGMA = 1.0		# The standard deviation of the well

center(t) = -im .* (RADIUS ./ 2) .* exp.( im .* (π./3) .* cos.(ω .* t))
wells(t) = [MvNormal([real(center(t)), imag(center(t))], SIGMA .* I(2))]
G(t) = MixtureModel(wells(t),[1]) |> FractionalNeuralSampling.Density

movie_saving_dir = plotsdir("SaccadeData/a=$(α_value)/b=$(β_value)/g=$(γ_value)")
data_saving_dir = datadir("SaccadeData/a=$(α_value)/b=$(β_value)/g=$(γ_value)")
mkpath(movie_saving_dir)
# mkpath(data_saving_dir)

seeds = [3202, 1600, 1927, 800, 2, 5]

for seedvalue in seeds

	time = Dates.format(now(), "HH_MM")
    local local_params = @strdict time α_value β_value γ_value seedvalue RADIUS ω SIGMA TIMESPAN

	println("Constructing Problem with Parameters: ")

    for (parameter, value) in local_params
        println(parameter, " => ", value, " ")
    end
	println("")

	L = dFNS(;
								u0 = vcat(x0, p0),  # Initial position and momentum changed to vcat from ArrayPartition
								tspan = TIMESPAN,
								α = α_value, # Tail index
								β = β_value, # Momentum strength
								γ = γ_value, # Noise strength
								𝜋 = G, # The target distribution
								boundaries = PeriodicBox(-1.5*RADIUS .. 1.5*RADIUS, -1.5*RADIUS .. 1.5*RADIUS),
								seed = seedvalue)

    println("Solving Problem!")
    δt = 0.001
    sol = solve(L, EM(); dt=δt, saveat=δt)          # using δt to save once per timestep (hopefully)
    println("Problem Solved!")

    doubled_data = DataFrame(sol)                   # this could create an instance of the data at the beginning and end of each timestep 
    walkerdata = unique(doubled_data, :timestamp)   # in case it does, we tell the walkerdata to only save each unique timestamp!

    ## Save CSV timeseries
    data_file = joinpath(data_saving_dir, savename(local_params, "csv"; connector = "-"))
    safesave(data_file, walkerdata)
    println("Saving data to: ", data_file)


    ### ---- PLOTTING TO BE DISABLED FOR LARGE RUNS! ---- ###
    #=

    ## Create movie directory with DrWatson
    movie_file = joinpath(movie_saving_dir, savename(local_params, "mp4"; connector = "-"))
    println("Saving visualisation to: ", movie_file)
	# Prepare Observables for Plotting

	xs = range(-RADIUS, RADIUS, length = 1000)
	ys = range(-RADIUS*0.8, 0, length = 1000)
	Xs = collect.(Iterators.product(xs, ys))
	X = Observable(G(0).(Xs)) # For recording
	xy = Observable(Point2f[]) # Initialize empty vector of points
	push!(xy[], Point2f(NaN, NaN)) 
	xy_last = Observable(Point2f([NaN, NaN]))

	trace_t = Observable([sol.t[1]])
    trace_x = Observable(reshape([sol[1,1]], 1, 1))


    # --- Figure Layout ---
    fig = Figure(size=(1000, 800))
    
    ax_main = Axis(fig[1, 1]; aspect=DataAspect(), title="Pendulum Tracking")
    ax_trace_x = Axis(fig[2, 1]; 
        ylabel = "Time", xlabel = "Position X",
        limits = ((-RADIUS, RADIUS), (0.0, TIMESPAN))
	)

    # colsize!(fig.layout, 1, Relative(0.75))
    # rowsize!(fig.layout, 1, Relative(0.75))

    # --- Plotting ---
    TimeseriesMakie.trail!(ax_main, xy; n_points=200, colormap=:Purples, color=1:100)
    TimeseriesMakie.heatmap!(ax_main, xs, ys, X, colormap=seethrough(:turbo))
    # TimeseriesMakie.scatter!(ax_main, xy_last; markersize=15, color=:red)
    hidedecorations!(ax_main); hidespines!(ax_main)

    plot_trace_x = TimeseriesMakie.traces!(ax_trace_x, trace_t, [1], trace_x; linecolor=:blue, spacing=0)

	Makie.rotate!(plot_trace_x, pi/2)
    
    record(fig, movie_file, range(1, length(sol), step=100); dpi=300, framerate=48) do i
        t = sol.t[i]
        u = sol[:, i]

        # Update Physics
		# Potential
		X[] = G(t).(Xs)
        xy[] = push!(xy[], Point2f(u[1:2]))
        length(xy[]) > 100 && popfirst!(xy[])
		notify(xy)

        xy_last[] = Point2f(u[1], u[2])

        # Update Traces
		current_indices = range(1, i, step=100)
		trace_t[] = sol.t[current_indices]
		current_vals = sol[1, current_indices]
        trace_x[] = -reshape(current_vals, length(current_vals), 1)  
    end

    =# 
    ### ---- END OF PLOTTING ---- ###
end
