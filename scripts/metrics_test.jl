using DrWatson
@quickactivate :VisualFNS

using FFTW 
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

const δt = 0.0001
xmax = 2000.
x0 = [3.0, 0.0]
p0 = [0.0, 0.0] 

G(t) = make_mixture(nx=16, ny=16, spacing=110.0, noise_level=60.0)(t)

function gaussian_kernel_function(grid_pos, μ, s)
    dist_sq = (grid_pos[1] - μ[1])^2 + (grid_pos[2] - μ[2])^2
    return exp(-dist_sq / (2 * (s^2))) 
end

function AdaptiveSolver(α_value, β_value, γ_value, timespan, seedvalue, sd, A, τ, approx_n_modes=128; n_proportion=0.02, target_radius=50.0)

    gaussian_kernel(grid_pos, μ) = gaussian_kernel_function(grid_pos, μ, sd)
    time_str = Dates.format(now(), "HH_MM")
    local local_params = @strdict time_str α_value β_value γ_value seedvalue A τ sd
    τ_d = A * τ

    println("Constructing Problem with Parameters: ")
    for (parameter, value) in local_params
        print(parameter, " => ", value, " ")
    end

    # --- 1. Target Selection Logic ---
    Random.seed!(seedvalue)
    nx, ny = 16, 16
    spacing = 110.0
    
    # Calculate the exact coordinates of the 256 well maxima
    wells_x = range(-(nx-1)/2 * spacing, step=spacing, length=nx)
    wells_y = range(-(ny-1)/2 * spacing, step=spacing, length=ny)
    all_wells = [[x, y] for x in wells_x, y in wells_y]
    
    # Randomly select n% of the wells (approx 5 targets)
    n_targets = max(1, round(Int, n_proportion * length(vec(all_wells))))
    target_points = shuffle(vec(all_wells))[1:n_targets]
    
    # We pass 0.05 directly as the metrics sampling rate
    L, saved_metrics, kernel_hist, grid = bayFNS(;         
        theta_dynamics=true,    
        α=α_value, 
        β=β_value, 
        γ=γ_value, 
        u0=vcat(x0, p0),
        tspan=timespan,
        𝜋=G,
        τ=τ,            
        τ_d=τ_d,            
        kernel=gaussian_kernel,
        
        metrics_saveat = 0.05,   # MACRO-STEP SCALE
        matrix_saveat = 0.05,  
        target_points = target_points, 
        target_radius = target_radius,
        
        approx_n_modes=approx_n_modes, 
        domain_length = 2.0 * xmax, 
        seed=seedvalue,
        boundaries = PeriodicBox(-xmax .. xmax, -xmax .. xmax)
    )

    println("\nSolving Problem!")
    sol = solve(L, EM(); dt=δt, saveat=δt)          
    println("Problem Solved!")

    # =====================================================================
    # Extract Metrics & Plot Static Dashboard 
    # =====================================================================
    fpt = (sol.t[end] < timespan) ? sol.t[end] : Inf
    
    t_trunc = saved_metrics.t
    kl_cum = [m[1] for m in saved_metrics.saveval]  
    kl_step = [m[2] for m in saved_metrics.saveval] 
    ent_trunc = [m[3] for m in saved_metrics.saveval]

    auec = trapz(t_trunc, ent_trunc)
    
    total_kl = sum(kl_step)
    total_time = t_trunc[end] - t_trunc[1]
    mean_info_vel = (total_time > 0) ? (total_kl / total_time) : 0.0

    if total_kl == 0
        gini_coeff = 1.0
        sorted_kl = [0.0]
    else
        sorted_kl = sort(kl_step)
        n = length(sorted_kl)
        index_sum = sum(i * sorted_kl[i] for i in 1:n)
        gini_coeff = (2.0 * index_sum) / (n * total_kl) - (n + 1.0) / n
    end

    fig_metrics = Figure(size=(1000, 800))
    
    ax_ent = Axis(fig_metrics[1, 1], title="Field Entropy (AUEC)", xlabel="Time", ylabel="Bits")
    lines!(ax_ent, t_trunc, ent_trunc, color=:purple, linewidth=2)
    
    ax_cum = Axis(fig_metrics[1, 2], title="Total Information Gained", xlabel="Time", ylabel="Cumulative KL")
    lines!(ax_cum, t_trunc, kl_cum, color=:orange, linewidth=2)
    band!(ax_cum, t_trunc, zeros(length(t_trunc)), kl_cum, color=(:orange, 0.2))

    ax_vel = Axis(fig_metrics[2, 1], title="Macro-Step KL-Divergence", xlabel="Time", ylabel="Information Gain")
    lines!(ax_vel, t_trunc, kl_step, color=:blue, linewidth=1.5)

    ax_gini = Axis(fig_metrics[2, 2], title="Lorenz Curve (Gini = $(round(gini_coeff, digits=2)))", 
                   xlabel="Cumulative Time Steps (Fraction)", ylabel="Cumulative Information Gain (Fraction)")
    n_kl = length(sorted_kl)
    cum_time_frac = collect(0:n_kl) ./ n_kl
    cum_info_frac = vcat([0.0], cumsum(sorted_kl) ./ total_kl)
    
    lines!(ax_gini, [0, 1], [0, 1], color=:black, linestyle=:dash)
    lines!(ax_gini, cum_time_frac, cum_info_frac, color=:red, linewidth=2)
    band!(ax_gini, cum_time_frac, cum_info_frac, cum_time_frac, color=(:red, 0.2))

    metrics_img_path = joinpath(datadir("Metrics-raw"), savename(local_params, "png"; connector = "-"))
    save(metrics_img_path, fig_metrics)
    # println("Saved metrics dashboard to: ", metrics_img_path)

    println("\nMetrics Summary:")
    println("  Total KL-Divergence: $total_kl")
    println("  Total Time: $total_time")
    println("  Mean Information Velocity: $mean_info_vel")
    println("  Gini Coefficient: $gini_coeff")

    metrics_img_path = joinpath(datadir("Metrics-raw"), savename(local_params, "png"; connector = "-"))
    save(metrics_img_path, fig_metrics)
    println("Saved metrics dashboard to: ", metrics_img_path)

    # =====================================================================
    # Export Summary CSV 
    # =====================================================================
    xs = [u[1] for u in sol.u]
    ys = [u[2] for u in sol.u]
    
    results_df = DataFrame(
        x = xs, y = ys, t = sol.t,
        alpha = α_value, beta = β_value, gamma = γ_value, tau = τ, seed = seedvalue,
        fpt = fpt, auec = auec,
        gini = gini_coeff, mean_info_velocity = mean_info_vel
    )
    
    data_path = datadir("Metrics-raw", savename(local_params, "csv"; connector = "-"))
    mkpath(datadir("Metrics-raw"))
    CSV.write(data_path, results_df)
    println("-> Run logged! FPT: $fpt | AUEC: $(round(auec, digits=2)) | Gini: $(round(gini_coeff, digits=3)) | Mean Info Velocity: $(round(mean_info_vel, digits=3))")

    # =====================================================================
    # 6. Render Animation (Updated Plotting)
    # =====================================================================
    t_min, t_max = extrema(sol.t)
    xlims_val = (-xmax, xmax)
    ylims_val = (-xmax, xmax)
    movie_dir = plotsdir("Metrics-movie")
    mkpath(movie_dir)
    movie_path = joinpath(movie_dir, savename(local_params, "mp4"; connector = "-"))
    println("Saving visualisation to: ", movie_path)

    x_axis = grid.xs
    y_axis = grid.ys
    n_grid = length(x_axis)

    frame_indices = 1:500:length(sol.t)
    n_frames = length(frame_indices)

    trace_t_data = fill(NaN, n_frames)
    trace_x_data = fill(NaN, n_frames, 1)
    trace_y_data = fill(NaN, n_frames, 1)

    trace_t = Observable(trace_t_data)
    trace_x = Observable(trace_x_data)
    trace_y = Observable(trace_y_data)

    K_pos_obs = Observable(zeros(Float64, n_grid, n_grid))

    xy = Observable([Point2f([NaN, NaN])])
    xy_last = Observable(Point2f([NaN, NaN]))

    fig = Figure(size=(1000, 800))

    ax_main = Axis(fig[1, 1]; aspect=DataAspect(), title="Adaptive Walk")
    ax_trace_x = Axis(fig[2, 1]; ylabel="X Position", xlabel="Time", limits = ((t_min, t_max), xlims_val))
    ax_trace_y = Axis(fig[1, 2]; ylabel="Time", xlabel="Y Position", limits = (ylims_val, (t_min, t_max)))

    colsize!(fig.layout, 1, Relative(0.75))
    rowsize!(fig.layout, 1, Relative(0.75))

    # --- Plotting Additions ---
    TimeseriesMakie.trail!(ax_main, xy; n_points=200, colormap=:Purples, color=1:100)
    TimeseriesMakie.contour!(ax_main, x_axis, y_axis, K_pos_obs, color=:orange, levels=5, linewidth=1.5, alpha = 0.6)

    # Plot the 5 randomly selected targets as blue crosses
    target_point_objects = [Point2f(tp...) for tp in target_points]
    TimeseriesMakie.scatter!(ax_main, target_point_objects; color=:blue, marker=:xcross, markersize=18)

    TimeseriesMakie.scatter!(ax_main, xy_last; markersize=15, color=:red)
    hidedecorations!(ax_main); hidespines!(ax_main)

    TimeseriesMakie.traces!(ax_trace_x, trace_t, [1], trace_x; linecolor=:blue, spacing=0)
    plot_trace_y = TimeseriesMakie.traces!(ax_trace_y, trace_t, [1], trace_y; linecolor=:red, spacing=0)

    Makie.rotate!(plot_trace_y, pi/2)

    last_k_idx = -1

    record(fig, movie_path, enumerate(frame_indices); framerate=48) do (frame_idx, i)
        t = sol.t[i]
        u = sol[:, i]
        xy[] = push!(xy[], Point2f(u[1:2]))
        length(xy[]) > 100 && popfirst!(xy[])
        xy_last[] = last(xy[])

        trace_t.val[frame_idx] = t
        trace_x.val[frame_idx, 1] = u[1]
        trace_y.val[frame_idx, 1] = u[2]

        notify(trace_t); notify(trace_x); notify(trace_y)
        k_idx = clamp(searchsortedfirst(kernel_hist.t, t), 1, length(kernel_hist.t))

        if k_idx != last_k_idx
            a_k_frame = kernel_hist.saveval[k_idx]
            K_vals_matrix = real.(FFTW.ifft(a_k_frame)) .* length(a_k_frame)

            @. K_pos_obs.val = max(K_vals_matrix, 0.0)

            notify(K_pos_obs)

            last_k_idx = k_idx
        end
    end
end

timespan = 50.
α_value = 1.2
β_value = 800.
γ_value = 300.
sd = 80.0
A = Inf
τ = 4e-5
approx_n_modes = 128

AdaptiveSolver(α_value, β_value, γ_value, timespan, 110, sd, A, τ, approx_n_modes)
