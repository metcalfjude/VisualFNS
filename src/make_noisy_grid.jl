
import FractionalNeuralSampling
using Distributions, LinearAlgebra, Random

export make_noisy_grid, make_mixture

function make_noisy_grid(; nx=4, ny=4, spacing=60.0, noise_level=10.0, cov_matrix=2.0*I(2), log_noise=0.04, seed=1272)
    Random.seed!(seed)
    # Center the nominal grid around (0,0)
    xs = range(-spacing * (nx - 1) / 2, spacing * (nx - 1) / 2, length=nx)
    ys = range(-spacing * (ny - 1) / 2, spacing * (ny - 1) / 2, length=ny)
    
    # Generate the wells by adding Gaussian noise to each grid point
    # Scale the covariance matrix by a log-normal factor to introduce variability in the spread of each well
    return [
        MvNormal([x + randn() * noise_level, y + randn() * noise_level], cov_matrix .* exp(randn() * log_noise)) 
        for x in xs for y in ys
    ]
end

function make_mixture(; nx=4, ny=4, spacing=60.0, noise_level=10.0, cov_matrix=2.0*I(2), log_noise=log(0.04), seed=1272)
    static_wells = make_noisy_grid(nx=nx, ny=ny, spacing=spacing, noise_level=noise_level, cov_matrix=cov_matrix, log_noise=log_noise, seed=seed)
    
    wells(t) = static_wells
    weights = fill(1.0 / length(static_wells), length(static_wells))
    
    density_func = t -> MixtureModel(wells(t), weights) |> FractionalNeuralSampling.Density
    
    # Extract the exact mean coordinates from the generated MvNormal components
    exact_centers = [mean(dist) for dist in static_wells] 
    
    return density_func, exact_centers
end
