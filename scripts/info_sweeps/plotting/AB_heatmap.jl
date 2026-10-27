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
sweep_folder = datadir("info_sweep-transition")
save_folder = "info_sweep_alpha_transition-raw" 
mkpath(plotsdir(save_folder))

# Set the target parameters, such that we can only include those data 
# matching our intended sweep, as there may be many files in the intended sweep folder. 
# This is especially important if there are multiple sweeps in the same folder.
# Locks the dataset so τ, sd, γ etc... are fixed β is free to sweep
target_params = Dict(
    "log_noise"     => 0.0,
    "noise_level"   => 20.0,
    "sd"            => 80.0,
    "target_radius" => 50.0,
    "well_size"     => 3,
    "τ"             => 0.0002, # Lock tau here! Adjust when we take a different slice
    "γ_value"       => 1000.0
)

all_files = readdir(sweep_folder, join=true)
files = filter(f -> endswith(f, ".csv") && !occursin("type=", f), all_files)

parsed = Tuple{Float64, Float64, String}[]

#  Parameter regex extraction
param_regex = r"([a-zA-Z_α-ωΑ-Ω]+)=([+-]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"

# Target β values for the sweep
# best to set this after we know the range of β values in the dataset, to avoid empty plots.
target_betas = 0:100:1200 # 800.:50.:1200.

# Values used for the one-dimensional slices. The nearest loaded value is used
# so these settings remain valid if the sweep grid changes slightly.
target_beta_slices = [0.0, 400.0, 800.0, 1200.0]
target_alpha_slices = [1.2, 1.4, 1.6, 1.8, 2.0]

for file in files
    fname = basename(file)
    params = Dict{String, Float64}()
    
    # Extract all key-value pairs into the dictionary
    for m in eachmatch(param_regex, fname)
        params[m.captures[1]] = parse(Float64, m.captures[2])
    end
    
    # Verify the file matches ALL target parameters exactly
    is_match = true
    for (k, v) in target_params
        if !haskey(params, k) || !isapprox(params[k], v, atol=1e-7)
            is_match = false
            break
        end
    end
    
    # If the file matches all target parameters, check for the presence of α and β values
    if is_match && haskey(params, "α_value") && haskey(params, "β_value")
        beta = params["β_value"]
        
        # ONLY accept files where beta matches one of our targets,
        # otherwise boxes may be double counting near-by β values from similar runs, shifting statistics
        if any(isapprox.(beta, target_betas, rtol=1e-3))
            push!(parsed, (params["α_value"], beta, file))
        end
    end
end

#Setup and initialise parameter arrays
alphas = sort(unique(x[1] for x in parsed))
betas = sort(unique(x[2] for x in parsed))
if !all(isapprox.(betas, target_betas, rtol=1e-3))
    error("Loaded beta values do not match the target_betas range. Loaded: $(betas), Target: $(target_betas)")
end

alpha_idx = Dict(a => i for (i, a) in enumerate(alphas))
beta_idx = Dict(b => i for (i, b) in enumerate(betas))
n_betas, n_alphas = length(betas), length(alphas)

# Define metrics of interest: (DataFrame Column Name => Output Filename Suffix)
metrics = [
    (:fpt, "fpt"),
    (:iv_mean, "mean_info_velocity"),
    (:iv_med, "median_info_velocity"),
    (:gini, "gini"),
    (:shape_auc, "shape_auc"),
    (:ig_entropy_grad, "ig_per_dent"),
    (:exponent_shape, "exponent_shape")
]

# Display names used in plot titles and axis labels. Edit these strings to
# change the wording shown on the figures without changing data columns or filenames.
metric_labels = Dict(
    :fpt => "FPT",
    :iv_mean => "Mean Information velocity",
    :iv_med => "IV",
    :gini => "Gini inequality",
    :shape_auc => "Shape area",
    :ig_entropy_grad => "IG per ΔEntropy",
    :exponent_shape => "SE",
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

# Initialise a dictionary to hold all matrices: 
# Keyed by (Column Symbol, Aggregation Function)
matrices = Dict((m[1], agg[1]) => fill(NaN, n_betas, n_alphas) for m in metrics, agg in aggs)
uncertainty_matrices = Dict(
    (m[1], agg[1]) => fill(NaN, n_betas, n_alphas)
    for m in metrics, agg in aggs
)

observations_store = Dict{Tuple{Symbol,Int,Int},Vector{Float64}}()   # (metric, β row, α col)

# Populate matrices
for (alpha, beta, file) in parsed
    df = CSV.read(file, DataFrame)
    filter!(:fpt => >(1), df)

    # Adding any post-hoc calculations
    df[!, :exponent_shape] = (1 .- df[!, :shape_auc]) ./ df[!, :shape_auc]
    
    r, c = beta_idx[beta], alpha_idx[alpha]
    
    for (col, _) in metrics
        observations = filter(isfinite, Float64.(df[!, col]))
        observations_store[(col, r, c)] = observations

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

# 3. Patch missing data (NaNs)
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

# Loop over all matrices and patch them
for mat in values(matrices)
    patch_nans_with_local_mean!(mat)
end

# Theory setup
x_idx, y_idx = 1:n_alphas, 1:n_betas
x_ticks = (x_idx, string.(alphas))
y_ticks = (y_idx, string.(betas))

γ = target_params["γ_value"]
σ_fix = target_params["well_size"]

theoretical_beta(alpha) = γ * (gamma(alpha - 1) / gamma(alpha / 2)^2) / (2 * σ_fix)
theory_beta = [theoretical_beta(a) for a in alphas]

function add_theory_contour!(p)
    plot!(p, alphas, theory_beta, color=:black, linewidth=2.5, linestyle=:dash, label="Periodic Boundary")
end

nearest_value(values, target) = values[argmin(abs.(values .- target))]

beta_slices = sort(unique(nearest_value(betas, beta) for beta in target_beta_slices))
alpha_slices = sort(unique(nearest_value(alphas, alpha) for alpha in target_alpha_slices))
beta_single_slice = nearest_value(betas, 1000.0)
alpha_single_slice = nearest_value(alphas, 1.2)

beta_colors = cgrad(:Oranges, length(beta_slices), categorical=true)
alpha_colors = reverse(cgrad(:Blues, length(alpha_slices), categorical=true))

# Master Function: Generate and Save Heatmaps Loop
for (col, file_suffix) in metrics
    for (agg_func, agg_name) in aggs
        mat = matrices[(col, agg_func)]
        
        p = heatmap(
            alphas, betas, mat,
            xticks=1.2:0.2:2.0, yticks=0:200:1200,
            xlabel="Alpha", ylabel="Beta",
            # title="$(agg_name) $(metric_labels[col])",
            c = :plasma
        )
        add_theory_contour!(p)
        
        # Automatically save with proper naming
        save_path = plotsdir(save_folder, "$(agg_name)_$(file_suffix)_heatmap.pdf")
        savefig(p, save_path)
    end
end

# Generate one-dimensional slices through each metric. Each figure contains
# several fixed-beta or fixed-alpha curves for direct comparison.
for (col, file_suffix) in metrics
    for (agg_func, agg_name) in aggs
        mat = matrices[(col, agg_func)]

        bootstrapped_correlation = 
        uncertainty = uncertainty_matrices[(col, agg_func)]

        beta_plot = plot(
            xlabel="α",
            ylabel=metric_labels[col],
            # title="$(agg_name) $(metric_labels[col]): fixed β",
        )
        for (series_idx, beta) in enumerate(beta_slices)
            plot!(
                beta_plot,
                alphas,
                mat[beta_idx[beta], :],
                ribbon=uncertainty[beta_idx[beta], :],
                color=beta_colors[series_idx],
                fillalpha=0.2,
                linewidth=2,
                marker=:circle,
                markersize=3,
                label="β = $(beta)",
            )
        end
        savefig(beta_plot, plotsdir(save_folder, "$(agg_name)_$(file_suffix)_vs_alpha.pdf"))

        alpha_plot = plot(
            xlabel="β",
            ylabel=metric_labels[col],
            # title="$(agg_name) $(metric_labels[col]): fixed α",
        )
        for (series_idx, alpha) in enumerate(alpha_slices)
            plot!(
                alpha_plot,
                betas,
                mat[:, alpha_idx[alpha]],
                ribbon=uncertainty[:, alpha_idx[alpha]],
                color=alpha_colors[series_idx],
                fillalpha=0.2,
                linewidth=2,
                marker=:circle,
                markersize=3,
                label="α = $(alpha)",
            )
        end
        savefig(alpha_plot, plotsdir(save_folder, "$(agg_name)_$(file_suffix)_vs_beta.pdf"))

        beta_single_plot = plot(
            legend=false,
            xlabel="α",
            ylabel=metric_labels[col],
            # title="$(agg_name) $(metric_labels[col]): β = $(beta_single_slice)",
            label="β = $(beta_single_slice)",
        )
        plot!(
            beta_single_plot,
            alphas,
            mat[beta_idx[beta_single_slice], :],
            ribbon=uncertainty[beta_idx[beta_single_slice], :],
            color=beta_colors[argmin(abs.(beta_slices .- beta_single_slice))],
            fillalpha=0.2,
            linewidth=2,
            marker=:circle,
            markersize=3,
        )
        savefig(
            beta_single_plot,
            plotsdir(save_folder, "$(agg_name)_$(file_suffix)_beta1000_vs_alpha.pdf"),
        )

        alpha_single_plot = plot(
            legend=false,
            xlabel="β",
            ylabel=metric_labels[col],
            # title="$(agg_name) $(metric_labels[col]): α = $(alpha_single_slice)",
            label="α = $(alpha_single_slice)",
        )
        plot!(
            alpha_single_plot,
            betas,
            mat[:, alpha_idx[alpha_single_slice]],
            ribbon=uncertainty[:, alpha_idx[alpha_single_slice]],
            color=alpha_colors[argmin(abs.(alpha_slices .- alpha_single_slice))],
            fillalpha=0.2,
            linewidth=2,
            marker=:circle,
            markersize=3,
        )
        savefig(
            alpha_single_plot,
            plotsdir(save_folder, "$(agg_name)_$(file_suffix)_alpha1p2_vs_beta.pdf"),
        )
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrapped rank correlations
#
# For a 1D slice (e.g. fixed β, sweeping α) we have, at each α, a set of
# independent per-run observations. One bootstrap replicate:
#   1. resample the runs at EACH α with replacement,
#   2. aggregate (median / mean) at each α  -> one value per α,
#   3. compute the rank correlation between α and those aggregated values.
# Repeating gives the sampling distribution of the correlation.
# Spearman and Kendall share the same resamples so they are directly comparable.
# Raw (unpatched) observations are used, so NaN-patching does not bias this.
# ─────────────────────────────────────────────────────────────────────────────

cor_methods = [(corspearman, "spearman"), (corkendall, "kendall")]
corr_rng = MersenneTwister(1929)   # separate RNG so results don't depend on earlier plotting

function bootstrap_rank_correlations(obs_per_point, x, agg_func, rng, n_bootstrap)
    # Need >1 observation at a point to resample meaningfully, and >=3 points for a correlation
    valid = findall(o -> length(o) > 1, obs_per_point)
    length(valid) >= 3 || return nothing

    xv = x[valid]
    ov = obs_per_point[valid]

    estimates = [cf(xv, agg_func.(ov)) for (cf, _) in cor_methods]   # point estimates, original data

    samples = fill(NaN, n_bootstrap, length(cor_methods))
    agg_vals = Vector{Float64}(undef, length(valid))
    for b in 1:n_bootstrap
        for (i, o) in enumerate(ov)
            agg_vals[i] = agg_func(rand(rng, o, length(o)))
        end
        for (j, (cf, _)) in enumerate(cor_methods)
            samples[b, j] = cf(xv, agg_vals)   # NaN if aggregated values are all tied
        end
    end
    return (; estimates, samples, n_points = length(valid))
end

function summarise_samples(samples)
    finite = filter(isfinite, samples)
    isempty(finite) && return (NaN, NaN, NaN, NaN, 0)
    lo, hi = quantile(finite, [0.025, 0.975])
    return (mean(finite), std(finite), lo, hi, length(finite))
end

correlation_rows = NamedTuple[]
bootstrap_samples = Dict{Tuple, Vector{Float64}}()   # raw distributions, kept in memory

# sweep_label: the variable being correlated against (x); fixed_label: the held-fixed variable
function run_sweep!(sweep_label, xvals, fixed_label, fixed_vals, get_obs)
    for fixed in fixed_vals, (col, _) in metrics, (agg_func, agg_name) in aggs
        res = bootstrap_rank_correlations(get_obs(col, fixed), xvals, agg_func,
                                          corr_rng, n_bootstrap)
        res === nothing && continue

        for (j, (_, method)) in enumerate(cor_methods)
            b_mean, b_sd, ci_lo, ci_hi, n_valid = summarise_samples(res.samples[:, j])
            push!(correlation_rows, (
                metric = String(col), agg = agg_name, method = method,
                sweep = sweep_label, fixed_param = fixed_label, fixed_value = fixed,
                estimate = res.estimates[j], boot_mean = b_mean, boot_sd = b_sd,
                ci_lo = ci_lo, ci_hi = ci_hi,
                n_points = res.n_points, n_valid_boot = n_valid,
            ))
            bootstrap_samples[(col, agg_name, method, sweep_label, fixed)] = res.samples[:, j]
        end
    end
end

# Correlation with α at fixed β (the original intent)
run_sweep!("alpha", alphas, "beta", beta_slices,
    (col, beta) -> [get(observations_store, (col, beta_idx[beta], c), Float64[])
                    for c in 1:n_alphas])

# Same thing the other way: correlation with β at fixed α
run_sweep!("beta", betas, "alpha", alpha_slices,
    (col, alpha) -> [get(observations_store, (col, r, alpha_idx[alpha]), Float64[])
                     for r in 1:n_betas])

correlation_df = DataFrame(correlation_rows)
CSV.write(plotsdir(save_folder, "bootstrap_rank_correlations.csv"), correlation_df)

# Summary plots: correlation (± 95% bootstrap CI) against the fixed parameter
for (col, file_suffix) in metrics, (_, agg_name) in aggs, sweep in ("alpha", "beta")
    sub = subset(correlation_df,
        :metric => ByRow(==(String(col))), :agg => ByRow(==(agg_name)),
        :sweep => ByRow(==(sweep)))
    isempty(sub) && continue

    fixed_sym = sweep == "alpha" ? "β" : "α"
    sweep_sym = sweep == "alpha" ? "α" : "β"
    p = plot(
        xlabel = "Fixed $(fixed_sym)", ylabel = "Rank correlation with $(sweep_sym)",
        # title = "$(agg_name) $(metric_labels[col])", ylims = (-1.05, 1.05), legend = :best,
    )
    hline!(p, [0.0], color = :gray, linestyle = :dash, label = "")
    for (_, method) in cor_methods
        m = subset(sub, :method => ByRow(==(method)))
        isempty(m) && continue
        scatter!(p, m.fixed_value, m.estimate;
            yerror = (m.estimate .- m.ci_lo, m.ci_hi .- m.estimate),
            label = method, markersize = 5)
    end
    savefig(p, plotsdir(save_folder, "$(agg_name)_$(file_suffix)_corr_vs_$(sweep_sym).pdf"))
end
