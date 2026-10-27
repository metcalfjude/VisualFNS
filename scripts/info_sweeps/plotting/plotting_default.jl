using Plots
# Set global publication defaults for Plots.jl
default(
    # Typography
    # titlefontsize = 0,     # No title by default for cleaner figures
    guidefontsize = 20,    # Axis label size
    tickfontsize = 15,     # Tick label size
    legendfontsize = 15,

    # Dimensions & Layout (e.g., standard single-column width)
    size = (450, 320),
    margin = 5Plots.mm,

    # Styling & Aesthetics
    framestyle = :box,    # Encloses the plot in a clean border
    grid = false,         # Turn off distracting background grids (or use true/alpha if needed)
    linewidth = 1.5,      # Thicker lines look sharper in print
    markersize = 4,
    palette = :tab10      # Professional, colorblind-friendly categorical palette
)
