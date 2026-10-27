using DrWatson
@quickactivate :VisualFNS

using Plots
using CSV
using DataFrames
using LinearAlgebra # For Euclidean distance calculations

# 1. Extract consecutive step lengths from the trajectory
function get_step_lengths(trajectory::Matrix{Float64})
    N = size(trajectory, 1)
    steps = zeros(N - 1)
    
    @inbounds for i in 1:(N - 1)
        # Distance between point i and i+1
        steps[i] = sqrt((trajectory[i+1, 1] - trajectory[i, 1])^2 + 
                        (trajectory[i+1, 2] - trajectory[i, 2])^2)
    end
    
    # Filter out stationary points (zeros) to avoid log(0) errors later
    return filter(x -> x > 0.0, steps)
end

println("Extracting step lengths...")
# Load your trajectory data
file = "/Users/chardiol/Desktop/Theory of Brain/Model_Fitting/data/ThetaAdap-raw/A=3000000-sd=80.0-seedvalue=27-time=16_05-α_value=1.6-β_value=650.0-γ_value=200.0-τ_d=0.6.csv"
data = CSV.read(file, DataFrame)

# Extract the relevant columns from the DataFrame
time = data.timestamp
x = data.value1
y = data.value2

trajectory = hcat(x, y)

# --- NEW SUBSAMPLING CODE ---
# Skip steps to coarse-grain the data. Adjust this number (e.g., 10, 100, 1000) 
# depending on how fine your SDE dt integration step is.
subsample_rate = 100
trajectory = trajectory[1:subsample_rate:end, :]
# ----------------------------

step_lengths = get_step_lengths(trajectory)

# 2. Calculate the CCDF (Rank-Frequency approach)
# We sort the steps from largest to smallest. 
# The rank divided by total N gives the probability of finding a step >= that length.
sorted_steps = sort(step_lengths, rev=true)
N_steps = length(sorted_steps)
ccdf_probs = (1:N_steps) ./ N_steps

# 3. Plot on a Log-Log scale
ccdf_plot = Plots.plot(
    sorted_steps, 
    ccdf_probs,
    seriestype = :scatter,
    xscale = :log10,
    yscale = :log10,
    markersize = 2,
    markerstrokewidth = 0, # Remove borders for a cleaner look with dense data
    color = :teal,
    alpha = 0.6,
    title = "Step Length CCDF",
    xlabel = "Step Length (l)",
    ylabel = "Probability P(L ≥ l)",
    label = "Gaze Data",
    legend = :bottomleft,
    grid = true
)

# Optional: Add a reference line to help eyeball the power-law slope
# If PDF slope is -μ, the CCDF slope will be -(μ - 1). 
# Example: If we assume a Lévy exponent μ = 2.0, the CCDF slope is -1.0.
l_ref = range(start=sorted_steps[round(Int, N_steps*0.8)], stop=sorted_steps[1], length=10)

# CHANGED: 10.0 replaced with 1e-3 (you may need to tweak this specific scalar 
# up or down so the line visually sits right on top of your data tail)
Plots.plot!(ccdf_plot, l_ref, l_ref .^ -2.0, 
            label="Theoretical Slope (μ = 3)", linestyle=:dash, color=:red, linewidth=2)
# Save the plot
savefig(ccdf_plot, plotsdir("Levy_CCDF_Plot.pdf"))
display(ccdf_plot)
