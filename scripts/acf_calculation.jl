using DrWatson
@quickactivate :VisualFNS

using FFTW
using Plots
using CSV
using DataFrames
using Statistics # For mean()

function calculate_acf_fft(trajectory::Matrix{Float64}, max_lags::Int)
    N, dims = size(trajectory)
    max_lags = min(max_lags, N - 1)
    
    acf = zeros(max_lags)
    
    # Pad to the next power of 2 greater than 2N to avoid circular 
    # correlation (wrap-around effects) and to optimize FFT speed.
    L = nextpow(2, 2 * N)
    
    for d in 1:dims
        x = trajectory[:, d]
        
        # Calculate the mean for this dimension
        μ = mean(x)
        
        # Mean-center the data and pad with zeros up to length L
        x_padded = zeros(L)
        x_padded[1:N] .= x .- μ
        
        # FFT, square the absolute values (power spectrum), and inverse FFT
        Fx = fft(x_padded)
        autocov = real.(ifft(abs2.(Fx))) # abs2.(Fx) is equivalent to Fx .* conj.(Fx)
        
        # The zero-lag autocovariance (variance) is at index 1
        var_x = autocov[1]
        
        # Extract the relevant lags (lag 1 is at index 2, etc.), 
        # normalize by variance, and average across dimensions
        for lag in 1:max_lags
            acf[lag] += (autocov[lag + 1] / var_x) / dims
        end
    end
    
    return acf
end

# --- Testing the performance ---

# Generate 50,000 steps of random walk
file = "/Users/chardiol/Desktop/Theory of Brain/Model_Fitting/data/ThetaAdap-raw/A=3000000-sd=80.0-seedvalue=27-time=01_02-α_value=1.6-β_value=650.0-γ_value=200.0-τ_d=0.6.csv"
data = CSV.read(file, DataFrame)

# Extract the relevant columns from the DataFrame
time = data.timestamp
x = data.value1
y = data.value2

# As mentioned before, if you want Velocity Autocorrelation (VACF) which 
# decays over time rather than Position Autocorrelation, use:
# trajectory = diff(hcat(x, y), dims=1)
trajectory = hcat(x, y)

max_steps = 100_000

println("Calculating ACF using FFT...")
@time acf_values_fft = calculate_acf_fft(trajectory, max_steps)

# Plot the results
acf_plot = Plots.plot(
    1:max_steps, 
    acf_values_fft,
    title  = "FFT-Accelerated Autocorrelation (ACF)",
    xlabel = "Time Lag (steps)",
    ylabel = "Autocorrelation",
    label  = "2D Random Walk (FFT)",
    linewidth = 2,
    xscale = :log10,
    color = :green,
    grid = true,
    legend = :topright
)

display(acf_plot)
