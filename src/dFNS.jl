import LabelledArrays
import FractionalNeuralSampling: divide_dims
import FractionalNeuralSampling.Boundaries: boundary_init
import LogDensityProblems

# * "Fractional neural sampling as a theory of spatiotemporal probabilistic computations in neural circuits", Qi and Gong
function dfns_f!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, β, γ = ps
    x, v = divide_dims(u, dimension(𝜋(t)))
    b = FractionalNeuralSampling.gradlogdensity(𝜋(t))(x) * gamma(α - 1) / (gamma(α / 2) .^ 2)
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= γ .* b .+ β .* v
    dv .= β .* b
end
function dfns_g!(du, u, p, t)
    ps, 𝜋 = p
    @unpack α, γ = ps
    dx, dv = divide_dims(du, dimension(𝜋(t)))
    dx .= γ^(1 / α) # ? × dL in the integrator.
    dv .= 0.0
end

function dFNS(;
    tspan, α, β, γ, u0=[0.0, 0.0],
    boundaries=nothing,
    noise_rate_prototype=similar(u0),
    𝜋=FractionalNeuralSampling.Density(default_density(first(u0))),
    noise=NoiseProcesses.LevyProcess!(α; ND=dimension(𝜋(0)),
        W0=zero(u0)),
    alg=EM(),
    callback=(),
    kwargs...)
    p = LabelledArrays.SLVector(; α, β, γ)

    Sampler(dfns_f!, dfns_g!;
        callback=CallbackSet(boundary_init(boundaries), callback...),
        kwargs...,
        u0,
        noise_rate_prototype, noise,
        tspan, p, 𝜋, alg) |> FractionalNeuralSampling.Samplers.assert_dimension(; order=2)
end

export dFNS
