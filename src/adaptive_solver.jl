### NOTE:
### When saving these files for parameter sweeps, make sure to queue the .csv files using Threads.@spawn
### batches, as the otherwise very frequent IO of FFT results especially will kill parallel jobs

function AdaptiveSolver(α_value, β_value, γ_value, timespan, seedvalue, sd, A, τ, approx_n_modes=128; 
    well_size=3, n_proportion=0.02, target_radius=50.0, noise_level=20.0, log_noise=log(0.04),
    x0 = [3.0, 0.0], p0 = [0.0, 0.0], xmax = 2000., δt = 0.0001,
    csv_folder = "info_sweep-raw", movie_folder = "info_sweep-movie",
    plot_lock = ReentrantLock()) # <-- Passing the lock as a kwarg to avoid global state issues

    # FFT parameters
    fs = 1/δt
    window_length_samples = round(Int, 3.0 * fs) 
    overlap_samples = div(window_length_samples, 2) 

    # Naming strings
    time_str = Dates.format(now(), "HH_MM")
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
        metrics_saveat = 0.05, 
        matrix_saveat = 0.05,  
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

    # =====================================================================
    # Extracting Information Metrics
    # =====================================================================

    # First passage time
    fpt = (sol.t[end] < timespan) ? sol.t[end] : timespan
    
    # Extract native metrics gathered by the bayFNS callback
    t_trunc = saved_metrics.t
    kl_cum = [m[1] for m in saved_metrics.saveval]  # Information Gain IG(t)
    kl_step = [m[2] for m in saved_metrics.saveval] # Discrete Info Gain Steps
    ent_trunc = [m[3] for m in saved_metrics.saveval] # Field Entropy H(t)
    
    # --- METRIC: Shape Parameter (Normalized Area Under Curve) ---
    # Normalize time and IG bounds to [0, 1] to decouple efficiency from speed
    t_norm = (fpt > 0) ? (t_trunc ./ fpt) : zeros(length(t_trunc))
    max_ig = maximum(kl_cum)
    ig_norm = (max_ig > eps()) ? (kl_cum ./ max_ig) : zeros(length(kl_cum))
    shape_auc = trapz(t_norm, ig_norm)

    # --- METRIC: Information Velocity (Mean & Median) ---
    dt_vals = diff(t_trunc)
    if length(dt_vals) > 0
        iv_array = kl_step[2:end] ./ dt_vals
        iv_mean = mean(iv_array)
        iv_med = median(iv_array)
    else
        iv_mean = 0.0
        iv_med = 0.0
    end
    
    # --- METRIC: Gini Coefficient (Inequality of Information Steps) ---
    total_kl = sum(kl_step)
    if total_kl <= eps()
        gini_coeff = 0.0
    else
        sorted_kl = sort(kl_step)
        n = length(sorted_kl)
        index_sum = sum(i * sorted_kl[i] for i in 1:n)
        gini_coeff = (2.0 * index_sum) / (n * total_kl) - (n + 1.0) / n
    end

    # --- METRIC: Information-Entropy Gradient ---
    dent = ent_trunc[1] - ent_trunc[end]
    ig_entropy_grad = (abs(dent) > eps()) ? (total_kl / dent) : 0.0
    
    # Store Summary Scalars
    results_df = DataFrame(
        alpha = α_value, beta = β_value, gamma = γ_value, tau = τ, seed = seedvalue,
        fpt = fpt, shape_auc = shape_auc, gini = gini_coeff, 
        iv_mean = iv_mean, iv_med = iv_med, ig_entropy_grad = ig_entropy_grad,
        max_ig = max_ig
    )

    local summary_params = @strdict α_value β_value γ_value τ well_size noise_level log_noise target_radius sd
    data_path = savepath(datadir, folder_name, summary_params, "csv")
    
    println("-> Run completed! Seed: $seedvalue | FPT: $(round(fpt, digits=2)) | Shape AUC: $(round(shape_auc, digits=2)) | Gini: $(round(gini_coeff, digits=2)) | Med IV: $(round(iv_med, digits=3))")
    
    # =====================================================================
    # POST-HOC REFORMULATION: Full Trajectory Extraction
    # =====================================================================
    timeseries_df = DataFrame(
        alpha = fill(α_value, length(t_trunc)),
        tau = fill(τ, length(t_trunc)),
        seed = fill(seedvalue, length(t_trunc)),
        time = t_trunc,
        time_norm = t_norm,
        ig_cum = kl_cum,
        ig_step = kl_step,
        entropy = ent_trunc
    )
    
    # Save timeseries using the exact same parameter grouping so files don't explode
    local ts_params = @strdict type="timeseries" α_value β_value γ_value τ well_size noise_level log_noise target_radius sd
    ts_data_path = savepath(datadir, folder_name, ts_params, "csv")
    
    # FFT logic for tracking Harmonic Gamma-Oscillations
    power_df = nothing
    power_data_path = ""
    freqs_out = nothing
    
    if length(x) > window_length_samples

        # Differentiate to "whiten" the 1/f power law, defined in src/VisualFNS.jl
        v_x = gradient(x, δt)
        v_y = gradient(y, δt)
        
        # Mean-zero
        v_x .-= mean(v_x)
        v_y .-= mean(v_y)
    
        # Find fixed resolution periodogram using Welch's Method
        pgram_x = welch_pgram(v_x, window_length_samples, overlap_samples; fs=fs)
        pgram_y = welch_pgram(v_y, window_length_samples, overlap_samples; fs=fs)

        # Gather results
        freqs_out = pgram_x.freq
        pxx = pgram_x.power
        pyy = pgram_y.power
        
        # Save results to a shared csv, and queue for saving 
        type = "power"
        local power_params = @strdict type α_value β_value γ_value τ well_size noise_level log_noise target_radius sd 
        power_data_path = savepath(datadir, folder_name, power_params, "csv")

        power_cols = ["alpha" => α_value, "beta" => β_value, "gamma" => γ_value, "tau" => τ, "seed" => seedvalue]
        for (i, p) in enumerate(pxx) push!(power_cols, "px_$i" => p) end
        for (i, p) in enumerate(pyy) push!(power_cols, "py_$i" => p) end

        power_df = DataFrame(power_cols)
    end

    # # =====================================================================
    # # Visual Inspection (Only for Seed 27)
    # # =====================================================================
    # if seedvalue == 27
    #     println("=> Seed 27 detected! Generating visual inspection plots and animation...")
    #     lock(plot_lock) do 

    #         type = "solution"
    #         local solution_params = @strdict type α_value β_value γ_value τ well_size noise_level log_noise target_radius sd
    #         true_path = joinpath(datadir(csv_folder), savename(solution_params, "csv"; connector = "-"))
    #         sol_df = DataFrame(t = sol.t, x = x, y = y, vx = sol[3, :], vy = sol[4, :])
    #         CSV.write(true_path, sol_df)

    #         # Static Dashboard
    #         fig_metrics = Figure(size=(1000, 800))
            
    #         ax_ent = Axis(fig_metrics[1, 1], title="Field Entropy (AUEC)", xlabel="Time", ylabel="Bits")
    #         lines!(ax_ent, t_trunc, ent_trunc, color=:purple, linewidth=2)
            
    #         ax_cum = Axis(fig_metrics[1, 2], title="Total Information Gained", xlabel="Time", ylabel="Cumulative KL")
    #         lines!(ax_cum, t_trunc, kl_cum, color=:orange, linewidth=2)
    #         band!(ax_cum, t_trunc, zeros(length(t_trunc)), kl_cum, color=(:orange, 0.2))
            
    #         ax_vel = Axis(fig_metrics[2, 1], title="Macro-Step KL-Divergence", xlabel="Time", ylabel="Information Gain")
    #         lines!(ax_vel, t_trunc, kl_step, color=:blue, linewidth=1.5)
            
    #         ax_gini = Axis(fig_metrics[2, 2], title="Lorenz Curve (Gini = $(round(gini_coeff, digits=2)))", 
    #         xlabel="Cumulative Time Steps (Fraction)", ylabel="Cumulative Information Gain (Fraction)")
    #         n_kl = length(sorted_kl)
    #         cum_time_frac = collect(0:n_kl) ./ n_kl
    #         cum_info_frac = vcat([0.0], cumsum(sorted_kl) ./ total_kl)
            
    #         lines!(ax_gini, [0, 1], [0, 1], color=:black, linestyle=:dash)
    #         lines!(ax_gini, cum_time_frac, cum_info_frac, color=:red, linewidth=2)
    #         band!(ax_gini, cum_time_frac, cum_info_frac, cum_time_frac, color=(:red, 0.2))
            
    #         metrics_img_path = joinpath(datadir(csv_folder), savename(local_params, "png"; connector = "-"))
    #         safesave(metrics_img_path, fig_metrics)
    #     end
    # end
    
    return (
        summary_path = data_path, 
        summary_df = results_df, 
        power_path = power_data_path, 
        power_df = power_df,
        ts_path = ts_data_path,
        timeseries_df = timeseries_df,
        freqs = freqs_out
    )
end

const infosolver_bFNS = AdaptiveSolver
