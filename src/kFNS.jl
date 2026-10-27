using LinearAlgebra
using ApproxFun
using DiffEqCallbacks
import LabelledArrays
import FractionalNeuralSampling
import FractionalNeuralSampling: divide_dims
import FractionalNeuralSampling.Boundaries: boundary_init
import LogDensityProblems
import LogDensityProblems: dimension
import StaticArraysCore
import StaticArrays

export kFNS

# Condition: Run every single time step
kernel_condition(u, t, integrator) = true

function kernel_effect!(integrator)
    ps = integrator.p[1]
    @unpack dim, sp, a_k, τ_d, τ_r, D_ops, base_coeffs, wavenumbers, ∇K_ref = ps
    u = integrator.u
    dt = integrator.dt

    x, _ = divide_dims(u, dim)

    # We iterate to update 'a_k' without creating new arrays
    if x isa Number
        # 1D Optimization
        val_scalar = x
        @inbounds for i in eachindex(a_k)
            dot_prod = wavenumbers[i, 1] * val_scalar
            
            shift_factor = cis(-dot_prod) 
            target_val = base_coeffs[i] * shift_factor
            
            # dy/dt = -y/τ_d + target/τ_r
            a_k[i] += dt * (-a_k[i] / τ_d + target_val / τ_r)
        end
    else
        # N-D Logic
        @inbounds for i in eachindex(a_k)
            dot_prod = 0.0
            for d in 1:dim
                dot_prod += wavenumbers[i, d] * x[d]
            end
            
            shift_factor = cis(-dot_prod) 
            target_val = base_coeffs[i] * shift_factor
            
            a_k[i] += dt * (-a_k[i] / τ_d + target_val / τ_r)
        end
    end
    
    # NOTE: ApproxFun Funs seem to like to be immutable. The fastest valid way 
    # without rewriting ApproxFun internals is to create the Fun once.
    K_new = Fun(sp, a_k)
    ∇K_ref[] = map(op -> op * K_new, D_ops)
    
    u_modified!(integrator, false) # Telling the solver that we are not changing `u` (efficiency)
end

function kfns_f!(du, u, p, t)
    ps_params, 𝜋 = p
    @unpack α, β, γ, ∇K_ref, sp, dim = ps_params

    x, v = divide_dims(u, dim)
    dx, dv = divide_dims(du, dim)

    b = time_robust_gradlogdensity(𝜋,t)(x) * gamma(α - 1) / (gamma(α / 2) .^ 2)
    
    # Access gradient functions
    ∇K_funcs = ∇K_ref[]

    if !(x isa Number)
         # Broadcast evaluation if x is a vector
         ∇K_vals = map(f -> real(f(x)), ∇K_funcs)
    else
         # Handle 1D scalar case
         ∇K_vals = map(f -> real(f(only(x))), ∇K_funcs)
    end

    k = (∇K_vals .* gamma(α - 1) ./ (gamma(α / 2) .^ 2))

    dx .= γ .* (b .- k) .+ β .* v
    dv .= β .* b
end

function kfns_g!(du, u, p, t)
    ps_params, _ = p
    @unpack α, γ, dim = ps_params
    
    dx, dv = divide_dims(du, dim)
    dx .= γ^(1 / α) 
    dv .= 0.0
end

function kFNS(;
        kernel, approx_n_modes,
        τ_r, τ_d,
        tspan, α, β, γ, u0=[0.0, 0.0],
        boundaries=nothing,
        noise_rate_prototype=similar(u0),
        𝜋=FractionalNeuralSampling.Density(default_density(first(u0))),
        noise=NoiseProcesses.LevyProcess!(α; ND=dimension(𝜋), W0=zero(u0)),
        alg=EM(),
        callback=(),
        kwargs...)

    dim = dimension(𝜋)

    # Use Laurent space for Shift Theorem logic
    sp_domain = Boundaries.domain(boundaries)
    sp = prod(Laurent.(sp_domain)) 

    # Define a complex kernel for Laurent
    kernel_complex(x) = Complex{Float64}(kernel(x))

    # We use a temporary Fun to guess the number of modes needed
    # We round using the approx number of modes provided by the user
    k_temp = Fun(kernel_complex, sp, round(Int, approx_n_modes))
    n_modes = length(k_temp.coefficients)
    
    # Generate the grid 
    grid_points = points(sp, n_modes)
    dummy_vals = zeros(ComplexF64, size(grid_points))
    plan = ApproxFunBase.plan_transform(sp, dummy_vals)

    # Calculate full number of modes (Plan might pad/truncate)
    transformed_dummy = plan * dummy_vals
    n_full_modes = length(transformed_dummy)

    # Calculate base coefficients (Shift Theorem logic)
    # We evaluate the kernel on the grid and transform it.
    base_vals = map(p -> kernel_complex(p), grid_points)
    base_coeffs = plan * base_vals 

    # Wavenumbers
    wavenumbers = zeros(Float64, (n_full_modes, dim)) 

    for d in 1:dim 
        # FIX: Create a vector of zeros with length 'dim'
        order_vec = zeros(Int, dim)
        order_vec[d] = 1
        
        # Now pass the dim dimensional vector, either [0, 1] or [1, 0]
        D_op = Derivative(sp, order_vec)
        
        # We perform the probe on the full size
        for i in 1:n_full_modes
            u_vec = zeros(ComplexF64, n_full_modes)
            u_vec[i] = 1.0
            
            # We use 'coefficients' directly if possible, or force size.
            f_basis = Fun(sp, u_vec)
            
            df = D_op * f_basis
            
            # Logic: D * exp(ikx) = (ik) * exp(ikx)
            val = (length(df.coefficients) >= i) ? df.coefficients[i] : 0.0
            wavenumbers[i, d] = imag(val)
        end
    end
    
    # Initialize Simulation Coefficients (a_k)
    a_k = copy(base_coeffs)

    # Setup gradient operators
    K_init = Fun(sp, a_k)
    Ds = StaticArrays.SVector{dim, Int}.(eachrow(I(dim)))
    if sp isa TensorSpace
        D_ops = Tuple(Derivative(sp, d) for d in Ds) 
    else
        Ds = only.(Ds)
        D_ops = Tuple(Derivative(sp, d) for d in Ds) 
    end

    ∇K_init = map(op -> op * K_init, D_ops)
    ∇K_ref = Ref(∇K_init)

    p = (; α, β, γ, τ_r, τ_d, dim,
        sp, 
        a_k,    
        ∇K_ref, 
        D_ops,
        base_coeffs, 
        wavenumbers
     )

    save_func = (u, t, integrator) -> copy(integrator.p[1].a_k)
    saved_a_k = SavedValues(Float64, Vector{ComplexF64})
    saver_cb = SavingCallback(save_func, saved_a_k, saveat=0.1) # Assuming max refresh rate for plotting of 10Hz is sufficient for simple visual diagnosis, can change with 'saveat'

    kernelcallback = DiscreteCallback(kernel_condition, kernel_effect!; save_positions=(false, false))
    full_cb = CallbackSet(boundary_init(boundaries), kernelcallback, saver_cb, callback...)

    sampler = Sampler(kfns_f!, kfns_g!;
            callback = full_cb,
            kwargs...,
            u0, noise_rate_prototype, noise, tspan,
            p, 𝜋, alg) |> FractionalNeuralSampling.Samplers.assert_dimension(; order=2)

    return sampler, saved_a_k, sp
end
