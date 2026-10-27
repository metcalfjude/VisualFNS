### NOTE:
### When saving these files for parameter sweeps, make sure to queue the .csv files using Threads.@spawn
### batches, as the otherwise very frequent IO of FFT results especially will kill parallel jobs

function AdaptiveSolverSave(α_value, β_value, γ_value, timespan, seedvalue, sd, A, τ, approx_n_modes=128; 
    well_size=3, n_proportion=0.02, target_radius=50.0, noise_level=20.0, log_noise=log(0.04),
    x0 = [3.0, 0.0], p0 = [0.0, 0.0], xmax = 2000., δt = 0.0001,
    csv_folder = "full-data", movie_folder = "full-movies",
    save_movie = (seedvalue % 27 == 0),
    plot_lock = ReentrantLock()) # <-- Passing the lock as a kwarg to avoid global state issues

    # FFT parameters
    fs = 1/δt
    window_length_samples = round(Int, 3.0 * fs) 
    overlap_samples = div(window_length_samples, 2) 

    # Naming strings
    time_str = Dates.format(now(), "HH_MM_SS")
    local local_params = @strdict time_str α_value β_value γ_value seedvalue A τ sd
    
    # Configuring information exploration and definition
    τ_d = A * τ         # The rate of kernel depletion, often if A=Inf the kernel doesn't decay at all, meaning no forgetting
    nx, ny = 16, 16     # Configuration of the dots being explored
    spacing = 110.0     # Average spacing of dots
    local_mixture_gen, exact_centers = make_mixture(nx=nx, ny=ny, spacing=spacing, cov_matrix=(well_size^2 .* I(2)), noise_level=noise_level, seed=seedvalue+1000, log_noise=log_noise)
    local_G(t) = local_mixture_gen(t)
    
    # Make sure to define the dots present, selecting randomly n_targets number as the target points
    n_targets = max(1, round(Int, n_proportion * length(exact_centers)))
    Random.seed!(seedvalue+2000)
    target_points = shuffle(exact_centers)[1:n_targets]
    
    # Setting up problem
    println("Constructing Problem (Seed: $seedvalue, α: $α_value, β: $β_value, γ: $γ_value, τ: $τ)")
    L, saved_metrics, kernel_hist, grid = bayFNS(;         
        theta_dynamics=true,    
        α=α_value, 
        β=β_value, 
        γ=γ_value, 
        u0=vcat(x0, p0),
        tspan=timespan,
        𝜋=local_G,
        τ=τ,
        τ_d=τ_d,
        sd=sd,
        metrics_saveat = 0.05,   # save metrics
        matrix_saveat = 0.05,  # save matrices
        target_points = target_points, 
        target_radius = target_radius,
        approx_n_modes=approx_n_modes, 
        domain_length = 2.0 * xmax, 
        seed=seedvalue,
        boundaries = PeriodicBox(-xmax .. xmax, -xmax .. xmax)
    )

    folder_name = csv_folder
    folder_dir = datadir(folder_name)
    mkpath(folder_dir)
    
    sol = solve(L, EM(); dt=δt, saveat=δt)
    print("Problem solved! (Seed: $seedvalue, α: $α_value, β: $β_value, γ: $γ_value, τ: $τ)")
    x, y = eachrow(sol[1:2, :])

    local data_params = @strdict α_value β_value γ_value τ well_size noise_level log_noise target_radius sd time_str seedvalue
    data_path = savepath(datadir, folder_name, data_params, "csv")

    results_df = DataFrame(t = sol.t, x = x, y = y)
    CSV.write(data_path, results_df)
    
    # =====================================================================
    # Visual Inspection (Only for Seeds divisible by 27 unless explicitly enabled)
    # =====================================================================
    if save_movie
        println("=> Generating visual inspection plots and animation...")
        lock(plot_lock) do 
            t_min, t_max = extrema(sol.t)
            dot_x = first.(exact_centers)
            dot_y = last.(exact_centers)

            dot_width  = maximum(dot_x) - minimum(dot_x)
            dot_height = maximum(dot_y) - minimum(dot_y)
            padding = 0.15 * max(dot_width, dot_height)

            dot_xlim = (
                minimum(dot_x) - padding,
                maximum(dot_x) + padding
            )

            dot_ylim = (
                minimum(dot_y) - padding,
                maximum(dot_y) + padding
            )
            movie_dir = plotsdir(movie_folder)
            mkpath(movie_dir)
            movie_path = joinpath(movie_dir, savename(local_params, "mp4"; connector = "-"))
            
            x_axis = grid.xs
            y_axis = grid.ys
            n_grid = length(x_axis)
            
            # COMPUTE THE STATIC HEATMAP (ORIGINAL PRIOR)
            prior_density = local_G(0.0)
            K_base = zeros(Float64, n_grid, n_grid)
            for (i, x_val) in enumerate(x_axis), (j, y_val) in enumerate(y_axis)
                K_base[i, j] = -log(prior_density([x_val, y_val]) + eps(0.0))
            end
            
            # Convert to normalized probability, then log10 scale
            M_base = minimum(K_base)
            P_base = exp.( -(K_base .- M_base) )
            P_base ./= maximum(P_base) 
            log_P_base = log10.(max.(P_base, 1e-6)) 

            frame_indices = unique(vcat(collect(1:250:length(sol.t)), length(sol.t)))
            n_frames = length(frame_indices)
            
            trace_t_data = fill(NaN, n_frames)
            trace_x_data = fill(NaN, n_frames, 1)
            
            trace_t = Observable(trace_t_data)
            trace_x = Observable(trace_x_data)

            # dynamic kernel observable for the contour
            K_pos_obs = Observable(zeros(Float64, n_grid, n_grid))

            xy = Observable([Point2f([NaN, NaN])])
            xy_last = Observable(Point2f([NaN, NaN]))

            fig = Figure(size=(1400, 1000))

            axis_style = (
                xticklabelsize = 44,
                yticklabelsize = 44,
                xticksize      = 24,
                yticksize      = 24,
                xlabelsize     = 42,
                ylabelsize     = 42,
                titlesize      = 46,
                spinewidth     = 8,
            )

            ax_main = Axis(
                fig[1, 1];
                aspect = DataAspect(),
                limits = ((-1000.0, 1000.0), dot_ylim),
                xlabel = "X Position",
                ylabel = "Y Position",
                axis_style...
            )

            ax_trace_x = Axis(
                fig[2, 1];
                ylabel = "X Position",
                xlabel = "Time",
                limits = ((t_min, t_max), dot_ylim),
                axis_style...
            )

            colsize!(fig.layout, 1, Relative(0.75))
            rowsize!(fig.layout, 1, Relative(0.75))
            rowgap!(fig.layout, 1, 0)

            # PLOT Layering
            Makie.heatmap!(ax_main, x_axis, y_axis, log_P_base, colormap=:viridis, alpha=0.5, colorrange=(-6.0, 0.0))

            # Layer 2: The Agent's Trail
            TimeseriesMakie.trail!(ax_main, xy; n_points=200, colormap=:Purples, color=1:100)
            
            # Layer 3: Dynamic Memory Contour (Restored)
            TimeseriesMakie.contour!(ax_main, x_axis, y_axis, K_pos_obs, color=:orange, levels=5, linewidth=1.5, alpha = 0.6)

            # Layer 4: Points (Original Colors)
            target_point_objects = [Point2f(tp...) for tp in target_points]
            TimeseriesMakie.scatter!(ax_main, target_point_objects; color=:blue, marker=:xcross, markersize=18)
            TimeseriesMakie.scatter!(ax_main, xy_last; markersize=15, color=:red)
            
            hidespines!(ax_main)

            TimeseriesMakie.traces!(ax_trace_x, trace_t, [1], trace_x; linecolor=:blue, spacing=0)

            last_k_idx = -1

            # ANIMATION LOOP
            record(fig, movie_path, enumerate(frame_indices); framerate=48) do (frame_idx, i)
                t = sol.t[i]
                u = sol[:, i]
                xy[] = push!(xy[], Point2f(u[1:2]))
                length(xy[]) > 100 && popfirst!(xy[])
                xy_last[] = last(xy[])

                trace_t.val[frame_idx] = t
                trace_x.val[frame_idx, 1] = u[1]

                notify(trace_t); notify(trace_x)
                
                k_idx = clamp(searchsortedfirst(kernel_hist.t, t), 1, length(kernel_hist.t))

                if k_idx != last_k_idx
                    a_k_frame = kernel_hist.saveval[k_idx]
                    
                    # Update just the dynamic memory contour
                    K_vals_matrix = real.(FFTW.ifft(a_k_frame)) .* length(a_k_frame)
                    @. K_pos_obs.val = max(K_vals_matrix, 0.0)
                    notify(K_pos_obs)
                    
                    last_k_idx = k_idx
                end
            end
            static_path = joinpath(
                movie_dir,
                savename(local_params, "pdf"; connector = "-")
            )

            Makie.save(static_path, fig, px_per_unit=1)
            println("=> Static publication figure saved to: ", static_path)
            println("=> Animation saved to: ", movie_path)
        end
    end
end

const savesolver_bFNS = AdaptiveSolverSave
