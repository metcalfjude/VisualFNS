# FulldFNS

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> VisualFNS

It is authored by Jude L. Metcalf.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "VisualFNS"
```
which auto-activate the project and enable local path handling from DrWatson.

## Dynamical visual FNS

This is an implementation of the Fractional Neural Sampling model on a 2D visual landscape. The purpose of the implementation is to study the performance of an accurate FNS model to track dynamically changing functional goals in a visual sampling context, including wwith the addition of a dynamical kernel which encodes a Bayesian sense of information adaptation in the explored landscape. The file where this is implemented is `src/kFNS.jl`. The base layer of SDE integration and solution is implemented through the fantastic `FractionalNeuralSampling.jl` project, authored by Brendan J. Harris.
