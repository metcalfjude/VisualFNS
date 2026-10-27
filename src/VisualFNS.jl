module VisualFNS

using DrWatson
@quickactivate "VisualFNS"

export aFNS, dFNS, kFNS, dsFNS, bayFNS
export HyperbolicSecant, MvSech, sechwells
export savepath, nearest_center
export gradient
export make_noisy_grid, make_mixture
export AdaptiveSolver, infosolver_bFNS
export AdaptiveSolverSave, savesolver_bFNS

using CairoMakie,
		 Foresight,
		 StaticArrays,
		 CSV,
		 Dates,
    	 DifferentialEquations,
		 DataFrames,
    	 Distributions,
		 DSP,
		 FFTW,
    	 FractionalNeuralSampling,
		 TimeseriesMakie,
		 LabelledArrays,
    	 LinearAlgebra,
		 StableDistributions,
		 SpecialFunctions,
		 Trapz,
		 Random,
		 DiffEqNoiseProcess,
		 LogDensityProblems
LogDensityProblems.dimension(f::Function) = 2 # LogDensityProblems.dimension(f(0))

# Set FFTW threads here so it runs correctly when the module is imported
function __init__()
    FFTW.set_num_threads(1)
end

time_robust_gradlogdensity(f::Function, t::Number) = FractionalNeuralSampling.gradlogdensity(f(t))
time_robust_gradlogdensity(f::DistributionDensity, _::Float64) = FractionalNeuralSampling.gradlogdensity(f)

import SpecialFunctions: gamma
import StaticArrays: SVector

function savepath(basefun, subdir, params, ext)
    dir = basefun(subdir)
    mkpath(dir)
    joinpath(dir, savename(params, ext; connector="-"))
end

function nearest_center(centers::Vector{SVector{2,Float64}}, walker::SVector{2,Float64})
	isempty(centers) && return 0 # Empty case!

	min_meas = Inf
	closest_idx::Int = 0

	@inbounds for i in eachindex(centers)
		dist_sq = sum(abs2, centers[i] - walker)

		if dist_sq < min_meas
			min_meas = dist_sq
			closest_idx = i
		end
	end
	return closest_idx
end

function gradient(v::AbstractVector, dx::Real=1.0)
    n = length(v)
    grad = similar(v)

    n < 2 && return grad

    grad[1] = (v[2] - v[1]) / dx
    grad[end] = (v[end] - v[end-1]) / dx

    # Using central difference
    @inbounds for i in 2:(n-1)
        grad[i] = (v[i+1] - v[i-1]) / (2 * dx)
    end

    return grad
end

include("aFNS.jl")
include("bayFNS.jl")
include("dFNS.jl")
include("dsFNS.jl")
include("kFNS.jl")
# include("2DbFNS.jl"))
include("PendulumParams.jl")
include("HyperbolicSecantDistr.jl")
include("make_noisy_grid.jl")
include("adaptive_solver.jl")
include("data_save_solver.jl")
include(srcdir("plotting_defaults.jl"))

end
