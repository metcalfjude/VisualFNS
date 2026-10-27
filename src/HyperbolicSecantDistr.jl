export HyperbolicSecant, MvSech, sechwells

using Distributions
using Statistics
import Random

struct HyperbolicSecant{L<:Real,S<:Real} <: ContinuousUnivariateDistribution
    loc::L
    scale::S

    function HyperbolicSecant(loc::L, scale::S) where {L<:Real,S<:Real}
        scale > zero(S) || throw(ArgumentError("scale must be positive"))
        new{L,S}(loc, scale)
    end
end

Distributions.params(d::HyperbolicSecant) = (d.loc, d.scale)

Distributions.pdf(d::HyperbolicSecant, x::Real) =
    (1 / pi) * sech((x - d.loc) / d.scale) / d.scale

function Distributions.logpdf(d::HyperbolicSecant, x::Real)
    z = (x-d.loc)/d.scale
    return -log(pi) - log(cosh(z)) - log(d.scale)
end

Distributions.cdf(d::HyperbolicSecant, x::Real) =
    (2/pi) * atan(exp((x - d.loc) / d.scale))

# function Distributions.quantile(d::HyperbolicSecant, p::Real)
#     0 ≤ p ≤ 1 || throw(DomainError(p, "p must lie in [0,1]"))
#     d.loc + d.scale*(2/pi)*log(tan(pi*p/2))
# end

function Base.rand(rng::Random.AbstractRNG, d::HyperbolicSecant)
    quantile(d, rand(rng))
end

Base.rand(d::HyperbolicSecant) = rand(Random.default_rng(), d)

Statistics.mean(d::HyperbolicSecant) = d.loc
Statistics.var(d::HyperbolicSecant) = d.scale^2

function MvSech(loc::Vector{L}, scale::Vector{S}) where {L<:Real,S<:Real}
    length(loc) == length(scale) ||
    throw(DimensionMismatch("loc and scale must have the same length"))

    product_distribution(HyperbolicSecant.(loc, scale))
end

function sechwells(locs::Vector{Vector{L}}, scales::Vector{Vector{S}}) where {L<:Real,S<:Real}
    length(locs) == length(scales) ||
    throw(DimensionMismatch("locs and scales must have the same length"))

    dists = MvSech.(locs, scales)

    return dists
end
