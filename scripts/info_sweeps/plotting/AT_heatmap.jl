
using DrWatson
@quickactivate :VisualFNS

include("plotting_default.jl")  # Load the default plotting settings

using CSV
using DataFrames
using Random
using Statistics
using StatsBase
using Plots
using SpecialFunctions: gamma

# Define directories for data and plots
sweep_folder = datadir("info_sweep-theta")
save_folder = plotsdir("info_sweep-theta")
mkpath(plotsdir(save_folder))

# Fixed parameters; τ is swept and β is held fixed
target_params = Dict(
    "log_noise"     => 0.0,
    "noise_level"   => 20.0,
    "sd"            => 80.0,
    "target_radius" => 50.0,
    "well_size"     => 3,
    "β_value"       => 1000.0,
    "γ_value"       => 1000.0
)

all_files = readdir(sweep_folder, join=true)
files = filter(f -> endswith(f, ".csv") && !occursin("type=", f), all_files)

parsed = Tuple{Float64, Float64, String}[]

# Parameter regex extraction, including Greek τ
param_regex = r"([a-zA-Z_α-ωΑ-ΩτΤ]+)=([+-]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"

# Target τ values
target_taus = logrange(5e-4, 1e-3, 20)

# Select a subset of τ values for the one-dimensional slices
target_theta_slices = target_taus[1:4:end]
target_alpha_slices = [1.2, 1.4, 1.6, 1.8, 2.0]

for file in files
    fname = basename(file)
    params = Dict{String, Float64}()

    for m in eachmatch(param_regex, fname)
        params[m.captures[1]] = parse(Float64, m.captures[2])
    end

    # Match all fixed parameters
    is_match = all(
        haskey(params, k) && isapprox(params[k], v; atol=1e-7)
        for (k, v) in target_params
    )

    if is_match && haskey(params, "α_value") && haskey(params, "τ")
        tau = params["τ"]

        if any(isapprox(tau, t; rtol=1e-3, atol=1e-10) for t in target_taus)
            push!(parsed, (params["α_value"], tau, file))
        end
    end
end

isempty(parsed) && error(
    "No matching files found. Check target_taus, target_params, and filenames."
)

# Initialise parameter arrays
alphas = sort(unique(x[1] for x in parsed))
taus = sort(unique(x[2] for x in parsed))

if !all(any(isapprox(t, target; rtol=1e-3, atol=1e-10) for target in target_taus)
        for t in taus)
    error("Loaded τ values do not match target_taus. Loaded: $taus")
end

alpha_idx = Dict(a => i for (i, a) in enumerate(alphas))
tau_idx = Dict(t => i for (i, t) in enumerate(taus))

n_taus, n_alphas = length(taus), length(alphas)

# Define metrics of interest
metrics = [
    (:fpt, "fpt"),
    (:iv_mean, "mean_info_velocity"),
    (:iv_med, "median_info_velocity"),
    (:gini, "gini"),
    (:shape_auc, "shape_auc"),
    (:ig_entropy_grad, "ig_per_dent"),
    (:exponent_shape, "exponent_shape")
]

metric_labels = Dict(
    :fpt => "First passage time",
    :iv_mean => "Mean Information velocity",
    :iv_med => "Median Information velocity",
    :gini => "Gini inequality",
    :shape_auc => "Shape area",
    :ig_entropy_grad => "IG per ΔEntropy",
    :exponent_shape => "Learning exponent"
)

aggs = [(median, "median"), (mean, "mean")]
n_bootstrap = 2000
bootstrap_rng = MersenneTwister(1929)

function bootstrap_median_sem(observations, rng, n_bootstrap)
    n = length(observations)
    n > 1 || return NaN

    bootstrap_medians = [
        median(rand(rng, observations, n))
        for _ in 1:n_bootstrap
    ]
    return std(bootstrap_medians)
end

# Matrices are indexed by (τ index, α index)
matrices = Dict(
    (m[1], agg[1]) => fill(NaN, n_taus, n_alphas)
    for m in metrics, agg in aggs
)

uncertainty_matrices = Dict(
    (m[1], agg[1]) => fill(NaN, n_taus, n_alphas)
    for m in metrics, agg in aggs
)

# Populate matrices
for (alpha, tau, file) in parsed
    df = CSV.read(file, DataFrame)
    filter!(:fpt => >(1), df)

    df[!, :exponent_shape] = (1 .- df[!, :shape_auc]) ./ df[!, :shape_auc]


    r, c = tau_idx[tau], alpha_idx[alpha]

    for (col, _) in metrics
        observations = filter(isfinite, Float64.(df[!, col]))

        if !isempty(observations)
            matrices[(col, median)][r, c] = median(observations)
            matrices[(col, mean)][r, c] = mean(observations)
        end

        if length(observations) > 1
            uncertainty_matrices[(col, median)][r, c] =
                bootstrap_median_sem(observations, bootstrap_rng, n_bootstrap)

            uncertainty_matrices[(col, mean)][r, c] =
                std(observations) / sqrt(length(observations))
        end
    end
end

# Patch missing data with the local mean
function patch_nans_with_local_mean!(mat)
    rows, cols = size(mat)
    source = copy(mat)

    for r in 1:rows, c in 1:cols
        if isnan(mat[r, c])
            neighbors = Float64[]
            r > 1    && !isnan(source[r-1, c]) && push!(neighbors, source[r-1, c])
            r < rows && !isnan(source[r+1, c]) && push!(neighbors, source[r+1, c])
            c > 1    && !isnan(source[r, c-1]) && push!(neighbors, source[r, c-1])
            c < cols && !isnan(source[r, c+1]) && push!(neighbors, source[r, c+1])

            if !isempty(neighbors)
                mat[r, c] = mean(neighbors)
            end
        end
    end
end

for mat in values(matrices)
    patch_nans_with_local_mean!(mat)
end

# Transform τ into θ
σ_fix = target_params["well_size"]
σ_k = target_params["sd"]

theta_value(tau) = tau * (σ_k / σ_fix)^2
thetas = theta_value.(taus)

nearest_value(values, target) = values[argmin(abs.(values .- target))]

# Select actual loaded τ values, then convert them to θ for display
tau_slices = sort(unique(
    nearest_value(taus, tau) for tau in target_theta_slices
))
theta_slices = theta_value.(tau_slices)

alpha_slices = sort(unique(
    nearest_value(alphas, alpha) for alpha in target_alpha_slices
))

tau_single_slice = nearest_value(taus, 2e-4)
theta_single_slice = theta_value(tau_single_slice)
alpha_single_slice = nearest_value(alphas, 1.2)

# Colours for fixed-θ and fixed-α curves
theta_colors = cgrad(:Purples, length(tau_slices), categorical=true)
alpha_colors = reverse(cgrad(:Blues, length(alpha_slices), categorical=true))

# Heatmaps: α versus θ
for (col, file_suffix) in metrics
    for (agg_func, agg_name) in aggs
        mat = matrices[(col, agg_func)]

        p = heatmap(
            alphas, thetas, mat,
            xlabel="α",
            ylabel="θ",
            title="$(agg_name) $(metric_labels[col])",
            c=:plasma
        )

        savefig(
            p,
            plotsdir(save_folder, "$(agg_name)_$(file_suffix)_heatmap.pdf")
        )
    end
end

# One-dimensional slices
for (col, file_suffix) in metrics
    for (agg_func, agg_name) in aggs
        mat = matrices[(col, agg_func)]
        uncertainty = uncertainty_matrices[(col, agg_func)]

        # Fixed θ: metric versus α; each θ has its own colour
        theta_plot = plot(
            xlabel="α",
            ylabel=metric_labels[col],
            title="$(agg_name) $(metric_labels[col]): fixed θ",
            legend=:best
        )

        for (series_idx, tau) in enumerate(tau_slices)
            r = tau_idx[tau]
            theta = theta_value(tau)

            plot!(
                theta_plot,
                alphas,
                mat[r, :],
                ribbon=uncertainty[r, :],
                color=theta_colors[series_idx],
                fillalpha=0.2,
                linewidth=2,
                marker=:circle,
                markersize=3,
                label="θ = $(round(theta, sigdigits=4))"
            )
        end

        savefig(
            theta_plot,
            plotsdir(save_folder, "$(agg_name)_$(file_suffix)_vs_alpha_by_theta.pdf")
        )

        # Fixed α: metric versus θ; each α has its own colour
        alpha_plot = plot(
            xlabel="θ",
            ylabel=metric_labels[col],
            title="$(agg_name) $(metric_labels[col]): fixed α",
            legend=:best
        )

        for (series_idx, alpha) in enumerate(alpha_slices)
            c = alpha_idx[alpha]

            plot!(
                alpha_plot,
                thetas,
                mat[:, c],
                ribbon=uncertainty[:, c],
                color=alpha_colors[series_idx],
                fillalpha=0.2,
                linewidth=2,
                marker=:circle,
                markersize=3,
                label="α = $(alpha)"
            )
        end

        savefig(
            alpha_plot,
            plotsdir(save_folder, "$(agg_name)_$(file_suffix)_vs_theta_by_alpha.pdf")
        )

        # Single fixed-θ slice: metric versus α
        theta_single_plot = plot(
            xlabel="α",
            ylabel=metric_labels[col],
            title="$(agg_name) $(metric_labels[col]): θ = $(round(theta_single_slice, sigdigits=4))",
            legend=false
        )

        plot!(
            theta_single_plot,
            alphas,
            mat[tau_idx[tau_single_slice], :],
            ribbon=uncertainty[tau_idx[tau_single_slice], :],
            color=theta_colors[argmin(abs.(tau_slices .- tau_single_slice))],
            fillalpha=0.2,
            linewidth=2,
            marker=:circle,
            markersize=3
        )

        savefig(
            theta_single_plot,
            plotsdir(save_folder, "$(agg_name)_$(file_suffix)_theta_single_vs_alpha.pdf")
        )

        # Single fixed-α slice: metric versus θ
        alpha_single_plot = plot(
            xlabel="θ",
            ylabel=metric_labels[col],
            title="$(agg_name) $(metric_labels[col]): α = $(alpha_single_slice)",
            legend=false
        )

        plot!(
            alpha_single_plot,
            thetas,
            mat[:, alpha_idx[alpha_single_slice]],
            ribbon=uncertainty[:, alpha_idx[alpha_single_slice]],
            color=alpha_colors[argmin(abs.(alpha_slices .- alpha_single_slice))],
            fillalpha=0.2,
            linewidth=2,
            marker=:circle,
            markersize=3
        )

        savefig(
            alpha_single_plot,
            plotsdir(save_folder, "$(agg_name)_$(file_suffix)_alpha1p2_vs_theta.pdf")
        )
    end
end


function bootstrap_spearman(observations, alphas, rng, n_bootstrap, agg_func)
    n = length(observations)
    n > 1 || return NaN

    if agg_func == median
        bootstrap_data = median(rand(rng, observations, n))
    elseif agg_func == mean
        bootstrap_data = mean(rand(rng, observations, n))
    else
        throw(ArgumentError("Unsupported aggregation function"))
    end

    bootstrap_correlations = [
        corspearman(alphas, bootstrap_data)
        for _ in 1:n_bootstrap
    ]
    return bootstrap_correlations
end

function bootstrap_kendall(observations, alphas, rng, n_bootstrap, agg_func)
    n = length(observations)
    n > 1 || return NaN

    if agg_func == median
        bootstrap_data = median(rand(rng, observations, n))
    elseif agg_func == mean
        bootstrap_data = mean(rand(rng, observations, n))
    else
        throw(ArgumentError("Unsupported aggregation function"))
    end

    bootstrap_correlations = [
        corkendall(alphas, bootstrap_data)
        for _ in 1:n_bootstrap
    ]
    return bootstrap_correlations
end

alpha_spearman_correlations = Dict(
    (m[1], agg[1]) => fill(NaN, n_taus, n_bootstrap)
    for m in metrics, agg in aggs
)
alpha_kendall_correlations = Dict(
    (m[1], agg[1]) => fill(NaN, n_taus, n_bootstrap)
    for m in metrics, agg in aggs
)

for tau in target_theta_slices
    df = CSV.read(file, DataFrame)
    filter!(:fpt => >(1), df)

    df[!, :exponent_shape] = (1 .- df[!, :shape_auc]) ./ df[!, :shape_auc]

    r = tau_idx[tau]

    for (col, _) in metrics
        observations = filter(isfinite, Float64.(df[!, col]))

        if length(observations) > 1
            alpha_spearman_correlations[(col, median)][r,!] =
                bootstrap_spearman(observations, alphas, bootstrap_rng, n_bootstrap, median)
            alpha_kendall_correlations[(col, median)][r,!] =
                bootstrap_kendall(observations, alphas, bootstrap_rng, n_bootstrap, median)
            alpha_spearman_correlations[(col, mean)][r,!] =
                bootstrap_spearman(observations, alphas, bootstrap_rng, n_bootstrap, mean)
            alpha_kendall_correlations[(col, mean)][r,!] =
                bootstrap_kendall(observations, alphas, bootstrap_rng, n_bootstrap, mean)
        end
    end
end

CSV.write(
    plotsdir(save_folder, "alpha_spearman_correlations.csv"),
    DataFrame(alpha_spearman_correlations)
)
