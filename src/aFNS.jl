# import LabelledArrays
import StaticArrays
import FractionalNeuralSampling: divide_dims
import FractionalNeuralSampling.Boundaries: boundary_init
import LogDensityProblems

adaptive_condition(u, t, integrator) = true

function adaptive_effect!(integrator)
    ps = integrator.p[1]
    @unpack centers, adaps, τ = ps
    
    u = integrator.u
    dt = integrator.dt

    x, _ = divide_dims(u, 2) # x is the 2-vector of [x,y], 2 is the magic number for dimension 2

    closest_idx = nearest_center(centers,SVector{2,Float64}(x))

    adaps[closest_idx] -= (dt/τ) * adaps[closest_idx]

    for i in 1:num_wells
        if i == closest_idx
            continue # already iterated
        else
            adaps[i] += (dt/τ) * ( ( ((1 - adaps[closest_idx]) / (num_wells - 1))
                                    - adaps[i] )
                                    + adaps[closest_idx]/(num_wells-1))
        end
    end
    u_modified!(integrator, false) # Telling the solver that we are not changing `u` (efficiency)
end

# * "Fractional neural sampling as a theory of spatiotemporal probabilistic computations in neural circuits", Qi and Gong
function afns_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, β, γ, adaps = ps
    x, v = divide_dims(u, 2) # strictly 2 is the magic number for the dimension of π
    b = FractionalNeuralSampling.gradlogdensity(𝜋(adaps))(x) * gamma(α - 1) / (gamma(α / 2) .^ 2)
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= γ .* b .+ β .* v
    dv .= β .* b
end

function afns_g!(du, u, p, t)
    ps, _ = p
    @unpack α, γ = ps
    dx, dv = divide_dims(du, 2) # Strictly 2 is a magic number for dim(π)
    dx .= γ^(1 / α) # ? × dL in the integrator.
    dv .= 0.0
end

function aFNS(;
    tspan, α, β, γ, τ, u0=[0.0, 0.0],
    boundaries=nothing,
    noise_rate_prototype=similar(u0),
    𝜋=FractionalNeuralSampling.Density(default_density(first(u0))),
    centers=Vector{Vector{Float64}}[],
    noise=NoiseProcesses.LevyProcess!(α; ND=dimension(𝜋(ones(Float64,length(centers))./length(centers))),
        W0=zero(u0)),
    alg=EM(),
    callback=(),
    kwargs...)

    global num_wells = length(centers)

    centers = SVector{2,Float64}.(centers) # Convert to SVector for speed
    adaps = ones(Float64,num_wells)./num_wells # Create a unit vector of adaptive coefficients
    p = (; α, β, γ, centers, adaps, τ)

    save_func = (u, t, integrator) -> copy(integrator.p[1].adaps)
    saved_adaps = SavedValues(Float64, Vector{Float64})
    saver_cb = SavingCallback(save_func, saved_adaps, saveat=0.1) # Assuming max refresh rate for plotting of 10Hz is sufficient for simple visual diagnosis, can change with 'saveat'

    adaptivecallback = DiscreteCallback(adaptive_condition, adaptive_effect!; save_positions=(false, false))

    callbackset = CallbackSet(boundary_init(boundaries), adaptivecallback, saver_cb, callback...)

    sampler = Sampler(afns_f!, afns_g!;
        callback=callbackset,
        kwargs...,
        u0,
        noise_rate_prototype, noise,
        tspan, p, 𝜋, alg) |> FractionalNeuralSampling.Samplers.assert_dimension(; order=2)
        
    return sampler,  saved_adaps
end

export aFNS
