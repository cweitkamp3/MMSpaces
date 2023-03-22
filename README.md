# MMSpaces
This packages implements the Gromov-Wasserstein distance (approximated by a Frak-Wolfe algorithm), its polynomial time computable lower bounds (see Mémoli (2011) and Chowdhury and Mémoli (2019) for more information) and the statistical tools for the analysis of metric measure spaces (or more general measure networks) proposed in Mordant et al. (2023).

## Installation

This package is developed under Linux (and WSL) and depends on [ThrustSort.jl](https://github.com/tscode/ThrustSort.jl) as well as [Network Simplex](https://github.com/nbonneel/network_simplex). Please check that the requirements for these libraries are met (in particular, ensure that the executable `nvcc` can be found and that the thrust header files are available). To install, run

```julia
using Pkg; Pkg.add(url="https://github.com/cweitkamp3/MMSpaces")
```
In the following, we briefly illustrate the usage of the main functions in simple settings.
## Gromov-Wasserstein
First of all, we showcase the algorithm for approximating (local minima of) the Gromov-Wasserstein distance, which is a popular tool for various data analysis tasks. Consider the following example:
```julia
using MMSpaces
using Distributions
using Distances
using CUDA

m = 2500
x = 4*rand(m, 2)

rho = sqrt.(rand(m))
theta = rand(Uniform(0, 2*pi), m)
y = [rho .* cos.(theta)  rho .* sin.(theta)]


C1 = pairwise(Euclidean(),x',x')
C2 = pairwise(Euclidean(),y',y')
p = fill(1/m, m)
q = fill(1/m, m)

@time      val1,plan1 = gromov_wasserstein(C1, C2, p, q; cuda = false)
@CUDA.time val2,plan2 = gromov_wasserstein(C1, C2, p, q; cuda = true)
```
It is important to note that the optimization problem considered here is highly non-convex. In particular, this means that the usage of a GPU (activated with `cuda == true`), which transforms the data into Float32 arrays for performance reasons, can impact the results.

## Lower Bounds
Unfortunately, approximating the Gromov-Wasserstein distance via gradient descents has several drawbacks: It is computationally intensive and there is no guarantee that one ends up at a global optima. Hence, it has been proposed to work with efficiently computable lower bounds of said distance. Three of these lower bounds have been implemented in this package (see Mémoli (2011) and Chowdhury and Mémoli (2019) for the precise definitions). We start by presenting the usage of the First Lower Bound (FLB).
```julia
flb(C1, C2; cuda = false)
flb(C1, C2; cuda = true)
Euc_flb(x, y; cuda = true)
flb(C1, C2, p, q)

# Timing the functions
@time Euc_flb(x, y; cuda = false)

x2 = cu(x)    # Copy data to GPU for a fair comparison  
y2 = cu(y)
@CUDA.time Euc_flb(x2, y2; cuda = true)
```
While we do not obtain a coupling between the measure networks considered, we note that we obtain a comparable value in a fraction of the time.

We make a similar observation for the Second Lower Bound (SLB) in this setting.
```julia
slb(C1, C2; cuda = false)
slb(C1, C2; cuda = true)
Euc_slb(x, y; cuda = true)
slb(x, y, p, q)

# Timing the functions
@time Euc_slb(x, y; cuda = false)

x2 = cu(x)    # Copy data to GPU for a fair comparison  
y2 = cu(y)
@CUDA.time Euc_slb(x2, y2; cuda = true)
```

Finally, we come to the Third Lower Bound (TLB). Although this lower bound has the highest computational comlpexity of the ones presented here, it is always larger than the other two and returns a matching of the measure networks considered, which depending on the application and additional regularization used can be quite helpful for data interpretation. Still, the computation is significantly faster than that of the Gromov-Wasserstein distance.
```julia
val3,plan3 = tlb(C1, C2; cuda = false, plan = true)
val4,plan4 = tlb(C1, C2; cuda = true)
val5,plan5 = Euc_slb(x, y; cuda = true)
val6,plan6 = tlbtlb(x, y, p, q)

# Timing the functions
@time Euc_tlb(x, y; cuda = false)

x2 = cu(x)    # Copy data to GPU for a fair comparison  
y2 = cu(y)
@CUDA.time Euc_tlb(x, y; cuda = true)
```
## Tests
Another advantage of the lower bounds presented is the fact that they are statistically accessible. Hence, it is possible to construct asymptotic tests for equivalence testing as well as testing for relevant differences based on these lower blounds (see Mordant et al. (2023)). More precisely given independent samples from two measure networks (C₁, p) and (C₂, q), we can construct tests for

`H₀ = LB((C₁,p), (C₂, q)) > Δ` versus `H₁ = LB((C₁, p), (C₂, q)) ≤ Δ`  (Equivalence Testing)

and

`H₀ = LB((C₁,p), (C₂, q)) ≤ Δ` versus `H₁ = LB((C₁, p), (C₂, q)) > Δ`  (Testing for Relevant Differences),

where LB stands for either FLB, SLB or TLB. 

In order to showcase the differences beween the resulting tests, let us consider the subsequent example:
```julia
function spiral_dist(n,v)
    sig = 0.03
    r = rand(n)
    A = [r.*sin.(v*r)+sig*rand(Normal(), n) r.*cos.(v*r)+sig*rand(Normal(), n) ]
  end
  
x = spiral_dist(500,10)
y = spiral_dist(500,15)

delta = 0.001
@CUDA.time FLB_Test(x, y, delta, 450,  450; hypo ="rel_diff", cuda = true, number = 1000)

@CUDA.time SLB_Test(x, y, delta, 450,  450; hypo ="rel_diff", cuda = true, number = 1000)

@CUDA.time TLB_Test(x, y, delta, 450,  450; hypo ="rel_diff", cuda = true, number = 1000)
```
It was already mentioned, that TLB is the largest of the three lower bounds. As a consequence, the test based on TLB is the only one that reliably rejects the null hypothesis in this setting.

## References
F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli."The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.


