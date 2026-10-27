using LinearAlgebra
using DiffEqCallbacks
using FFTW 
using SpecialFunctions
import LabelledArrays
import FractionalNeuralSampling
import FractionalNeuralSampling: divide_dims
import FractionalNeuralSampling.Boundaries: boundary_init
import LogDensityProblems
import LogDensityProblems: dimension

export bayFNS

bay_kernel_condition(u, t, integrator) = true

function bay_kernel_effect!(integrator)
    ps = integrator.p[1]
    @unpack dim, a_k, τ, τ_d, xs, ys, spatial_vals, target_coeffs, fft_plan, σ, sd, kx_vec, ky_vec = ps
    dt = integrator.dt

    x, _ = divide_dims(integrator.u, dim)

    @inbounds for i in eachindex(xs)
        # Reusing phase_x/phase_y memory as temporary real arrays if needed, or make new ones
        kx_vec[i] = exp(-((xs[i] - x[1])^2) / (2 * sd^2))
        ky_vec[i] = exp(-((ys[i] - x[2])^2) / (2 * sd^2))
    end

    @inbounds for j in 1:length(ys)
        ky_val = ky_vec[j]
        for i in 1:length(xs)
            spatial_vals[i, j] = kx_vec[i] * ky_val
        end
    end
    
    target_coeffs .= spatial_vals
    mul!(target_coeffs, fft_plan, target_coeffs)       
    
    N_total = length(target_coeffs)
    @inbounds for i in eachindex(target_coeffs)
        target_coeffs[i] /= N_total
    end

    @inbounds for i in eachindex(a_k)
        a_k[i] += dt * (-a_k[i] / τ_d + target_coeffs[i] / τ)
    end
    
    u_modified!(integrator, false) 
end

function bayfns_f!(du, u, p, t)
    ps_params, 𝜋 = p
    @unpack α, β, γ, dim, a_k, wavenumbers, ∇K_vals, xs, ys, theta_dynamics, phase_x, phase_y = ps_params

    x, v = divide_dims(u, dim)
    dx, dv = divide_dims(du, dim)

    b = time_robust_gradlogdensity(𝜋,t)(x) * gamma(α - 1) / (gamma(α / 2) ^ 2)

    if theta_dynamics
        fill!(∇K_vals, 0.0)

        # 1. Compute 1D phases (256 cis calls instead of 16,384)
        @inbounds for i in eachindex(wavenumbers)
            phase_x[i] = cis(wavenumbers[i] * (x[1] - xs[1]))
            phase_y[i] = cis(wavenumbers[i] * (x[2] - ys[1]))
        end

        # 2. Multiply them in the 2D loop
        @inbounds for j in 1:size(a_k, 2)
            py = phase_y[j]
            for i in 1:size(a_k, 1)
                kx = wavenumbers[i]
                ky = wavenumbers[j]
                
                phase = phase_x[i] * py # Single complex multiplication!
                
                ∇K_vals[1] += real(im * kx * a_k[i, j] * phase)
                ∇K_vals[2] += real(im * ky * a_k[i, j] * phase)
            end
        end
    
        k = ∇K_vals .* (gamma(α - 1) / (gamma(α / 2) .^ 2))
    
        dx .= γ .* (b .- k) .+ β .* v
        dv .= β .* b
    else
        dx .= γ .* b .+ β .* v
        dv .= β .* b
    end
end

function bayfns_g!(du, u, p, t)
    ps_params, _ = p
    @unpack α, γ, dim = ps_params
    
    dx, dv = divide_dims(du, dim)
    dx .= γ^(1 / α) 
    dv .= 0.0
end

function gaussian_kernel_function(grid_pos, μ, s)
    dist_sq = (grid_pos[1] - μ[1])^2 + (grid_pos[2] - μ[2])^2
    return exp(-dist_sq / (2 * (s^2))) 
end

function bayFNS(;
        theta_dynamics::Bool = true,
        target_radius::Float64 = 5.0,
        target_points = [[0.0, 0.0]], 
        metrics_saveat::Float64 = 0.05, 
        matrix_saveat::Float64 = 0.05,
        sd = sd,
        σ = 1.0,        
        approx_n_modes = 256,
        domain_length = 20.0, 
        τ, τ_d,
        tspan, α, β, γ, u0=[0.0, 0.0],
        boundaries=nothing,
        noise_rate_prototype=similar(u0),
        𝜋=FractionalNeuralSampling.Density(default_density(first(u0))),
        noise=NoiseProcesses.LevyProcess!(α; ND=dimension(𝜋), W0=zero(u0)),
        alg=EM(),
        callback=(),
        kwargs...)

    dim = dimension(𝜋)
    dx_step = domain_length / approx_n_modes
    xs = collect(range(-domain_length/2, domain_length/2, length=approx_n_modes))
    ys = collect(range(-domain_length/2, domain_length/2, length=approx_n_modes))

    spatial_vals = zeros(ComplexF64, approx_n_modes, approx_n_modes)
    target_coeffs = zeros(ComplexF64, approx_n_modes, approx_n_modes)
    a_k = zeros(ComplexF64, approx_n_modes, approx_n_modes) 
    
    fft_plan = FFTW.plan_fft!(target_coeffs, flags=FFTW.ESTIMATE)
    freqs = FFTW.fftfreq(approx_n_modes, 1/dx_step)
    wavenumbers = 2 * π .* freqs
    ∇K_vals = zeros(Float64, dim)

    phase_x = zeros(ComplexF64, approx_n_modes)
    phase_y = zeros(ComplexF64, approx_n_modes)

    kx_vec = zeros(length(xs))
    ky_vec = zeros(length(ys))

    p = (; α, β, γ, dim, 
        a_k, wavenumbers, ∇K_vals, 
        xs, ys, spatial_vals, target_coeffs, fft_plan, 
        τ, τ_d, target_points, target_radius, σ,
        phase_x, phase_y, kx_vec, ky_vec,
        sd, theta_dynamics)

    # =======================================================
    # Construct the True Prior (Base Saliency)
    # =======================================================
    K_base = zeros(Float64, approx_n_modes, approx_n_modes)
    
    # Evaluate the time-dependent density at t = 0.0 to get the spatial Prior
    prior_density_landscape = 𝜋(0.0) 
    
    for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
        # Evaluate the spatial landscape at coordinate [x, y]
        density_val = prior_density_landscape([x, y])
        
        # Add a tiny epsilon to prevent log(0) in mathematically empty regions
        K_base[i, j] = -log(density_val + eps(0.)) 
    end
    
    M_prior = minimum(K_base)
    K_shifted_prior = K_base .- M_prior
    Z_prior = sum(exp.(-K_shifted_prior))
    log_Z_prior = log(Z_prior) - M_prior

    # =======================================================
    # Callback: Native Macro-Step Information Tracking
    # =======================================================
    K_prev_mem = zeros(Float64, approx_n_modes, approx_n_modes)
    L_prev = zeros(Float64, approx_n_modes, approx_n_modes)
    log_Z_prev = Ref(0.0)
    first_call = Ref(true) 

    metrics_save_func = (u, t, integrator) -> begin
        
        # PHYSICAL FPT TERMINATION
        x, _ = divide_dims(u, integrator.p[1].dim)
        hit_target = false
        for tp in integrator.p[1].target_points
            if sqrt((x[1] - tp[1])^2 + (x[2] - tp[2])^2) <= integrator.p[1].target_radius
                hit_target = true
                break
            end
        end
        
        if hit_target
            terminate!(integrator)
        end

        # Reconstruct Total Potential (Base + Memory)
        current_a_k = integrator.p[1].a_k
        K_mem_curr = real.(FFTW.ifft(current_a_k)) .* length(current_a_k)
        K_total_curr = K_base .+ K_mem_curr
        
        M_curr = minimum(K_total_curr)
        K_shifted_curr = K_total_curr .- M_curr
        exp_K_curr = exp.(-K_shifted_curr)
        Z_shifted_curr = sum(exp_K_curr)
        
        L_curr = exp_K_curr ./ Z_shifted_curr
        log_Z_curr = log(Z_shifted_curr) - M_curr 
        
        ent = -sum(L_curr .* log2.(L_curr .+ 1e-12))
        
        # --- METRIC 1: Cumulative KL (Prior to Posterior) ---
        kl_cum = sum(L_curr .* (-K_mem_curr)) + log_Z_prior - log_Z_curr
        
        if first_call[]
            K_prev_mem .= K_mem_curr
            L_prev .= L_curr
            log_Z_prev[] = log_Z_curr
            first_call[] = false
            return (0.0, 0.0, ent)
        end
        
        # --- METRIC 2: Macro-Step KL (Previous to Current) ---
        kl_step = sum(L_curr .* (K_prev_mem .- K_mem_curr)) + log_Z_prev[] - log_Z_curr
        
        K_prev_mem .= K_mem_curr
        L_prev .= L_curr
        log_Z_prev[] = log_Z_curr
        
        return (max(0.0, kl_cum), max(0.0, kl_step), ent)
    end

    saved_metrics = SavedValues(Float64, Tuple{Float64, Float64, Float64})
    metrics_saver_cb = SavingCallback(metrics_save_func, saved_metrics, saveat=metrics_saveat)

    ak_save_func = (u, t, integrator) -> copy(integrator.p[1].a_k)
    saved_a_k = SavedValues(Float64, Matrix{ComplexF64})
    ak_saver_cb = SavingCallback(ak_save_func, saved_a_k, saveat=matrix_saveat) 

    bay_kernelcallback = DiscreteCallback(bay_kernel_condition, bay_kernel_effect!; save_positions=(false, false))
    full_cb = CallbackSet(boundary_init(boundaries), bay_kernelcallback, metrics_saver_cb, ak_saver_cb, callback...)

    sampler = Sampler(bayfns_f!, bayfns_g!;
            callback = full_cb,
            kwargs...,
            u0, noise_rate_prototype, noise, tspan,
            p, 𝜋, alg) |> FractionalNeuralSampling.Samplers.assert_dimension(; order=2)

    return sampler, saved_metrics, saved_a_k, (; xs, ys)
end
