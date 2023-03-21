using LinearAlgebra
using Distances
using CUDA
using StatsBase
using ThrustSort


# We call functions from the following two shared libraries
const lib_emd_bonneel = joinpath(@__DIR__, "emd_bonneel/emd_bonneel.so")
# ---------------------------------------------------------------------------- #
# Auxiliary functions
# ---------------------------------------------------------------------------- #


safe_sqrt(a) = sqrt(max(a, 0))

"""
    ``gpu_helper_function(m,n)``

Calculates the transport plan of the "Earth Mover's distance" between two uniform distributions on the real line (with ``m`` and ``n`` points, respectively).
Returns GPU-arrays. This function is neccessary for the fast calculation of [`tlb_gpu`](@ref).
"""
function gpu_helper_function(m,n)
    mat_size = n+m-1
    mass_diff = zeros(mat_size)
    dupl_m = zeros(Int64,mat_size)
    dupl_n = zeros(Int64,mat_size)
    wm = 1.0/m
    wn = 1.0/n
    index1 = 1
    index2 = 1
    for i in 1:mat_size
      if(wn < wm)
        dupl_m[i]= index1
        dupl_n[i]= index2
        wm = wm - wn
        mass_diff[i] = wn
        wn = 1.0/n
        index2 += 1
      else
        dupl_m[i]= index1
        dupl_n[i]= index2
        wn = wn - wm
        mass_diff[i]=wm
        wm = 1.0/m
        index1 += 1
      end
    end
    return CuArray{Int32}(dupl_m), CuArray{Int32}(dupl_n), cu(mass_diff);
end

"""
    ``cpu_helper_function(m,n)``

Calculates the transport plan of the "Earth Mover's distance" between two uniform distributions on the real line (with ``m`` and ``n`` points, respectively).
Returns CPU-arrays. This function is neccessary for the fast calculation of [`tlb_cpu`](@ref).
"""
function cpu_helper_function(m,n)
  mat_size = n+m-1
  mass_diff = zeros(mat_size)
  dupl_m = zeros(Int64,mat_size)
  dupl_n = zeros(Int64,mat_size)
  wm = 1.0/m
  wn = 1.0/n
  index1 = 1
  index2 = 1
  for i in 1:mat_size
    if(wn < wm)
      dupl_m[i]= index1
      dupl_n[i]= index2
      wm = wm - wn
      mass_diff[i] = wn
      wn = 1.0/n
      index2 += 1
    else
      dupl_m[i]= index1
      dupl_n[i]= index2
      wn = wn - wm
      mass_diff[i]=wm
      wm = 1.0/m
      index1 += 1
    end
  end
  return Array{Int32}(dupl_m), Array{Int32}(dupl_n), (mass_diff);
end


# ---------------------------------------------------------------------------- #
# Optimal Transport
# ---------------------------------------------------------------------------- #

# 1d-Wasserstein

function Wasserstein1d_helper(x, y, p)
  m = length(x)
  n = length(y)
  res = 0

  i = 1
  w_i = 1/m

  j = 1
  w_j = 1/n
  while true
      m_ij = (abs(x[i] - y[j]))^p
      if w_i < w_j || j == n
          res += m_ij*w_i
          i += 1
          if i == m+1
              break
          end
          w_j -= w_i
          w_i = 1/m
      else
          res += m_ij*w_j
          j += 1
          if j == n+1
              break
          end
          w_i -= w_j
          w_j = 1/n
      end
  end
  return res
end

"""
    ``Wasserstein1d(x, y, wx, wy, p)``

Calculate the "Earth Mover's distance" of order ``p`` on the real line, where the sorted vectors ``x`` and ``y``
describe the locations and the corresponding measures are assumed to be uniform distributions.
"""
function Wasserstein1d(x, y, p)

  m = length(x)
  n = length(y)

  if m==n
      res = (sum(x->abs(x)^p, x-y)*1.0/n)
  else     
      res = Wasserstein1d_helper(x, y, p)
  end
  
  return res
  
end
"""
    ``Wasserstein1d(x, y, wx, wy, p)``

Calculate the "Earth Mover's distance" of order ``p`` on the real line, where the sorted vectorw ``x`` and ``y``
describe the locations and the probability vectors ``wx`` and ``wy`` encode the mass at each location.
"""
function Wasserstein1d(x, y, wx, wy, p)
  m = length(x)
  n = length(y)
  res = 0

  i = 1
  w_i = wx[i]

  j = 1
  w_j = wy[j]
  while true
      m_ij = (x[i] - y[j]) * (x[i] - y[j])
      if w_i < w_j || j == n
          res += m_ij*w_i
          i += 1
          if i == m+1
              break
          end
          w_j -= w_i
          w_i = wx[i]
      else
          res += m_ij*w_j
          j += 1
          if j == n+1
              break
          end
          w_i -= w_j
          w_j = wy[j]
      end
  end
  return res
end

# Interface to Bonneel-Code

"""
    ``emd_bonneel(a, b, c; max_iter = 10_000_000)``

Calculate the "Earth Mover's distance" between
probability vectors `a` and `b` under cost matrix `c`.  Currently, only arrays
with data type `Float64` are supported.

# Arguments
- `a::Vector{Float64}`: Probability vector of length ``m``.
- `b::Vector{Float64}`: Probability vector of length ``n``.
- `c::Matrix{Float64}`: Cost matrix of dimension ``m x n``
- `max_iter::Int64`: Maximal number of iterations.

# Returns
- `res` : The value of "Earth Mover's distance" between `a` and `b` under cost `c`.

Throws a warning if the problem is unbounded, is infeasible, or if the maximal
number of iterations has been reached. See also [`emd_bonneel_with_plan`](@ref).
"""
function emd_bonneel(a :: Vector{Float64}, b :: Vector{Float64}, c :: Matrix{Float64}; max_iter :: Int = 10_000_000)
  if (length(a) != size(c)[1] || length(b) != size(c)[2] )
    error("The length of the weight vectors does not match the dimensions of the cost matrix")
  end
  max_iter = max(0, max_iter)
  res = ccall( (:emd_bonneel, lib_emd_bonneel)
             , Tuple{Float64, Int}
             , (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Int64, Csize_t)
             , a, b, c, length(a), length(b), max_iter)
  if res[2] == 1
    @warn "problem unbounded"
  elseif res[2] == 2 && (!isapprox(sum(a),1) ||  !isapprox(sum(b),1))
    @warn "problem infeasible. Check that a and b are in the simplex"
  elseif res[2] == 2 && (isapprox(sum(a),1) &&  isapprox(sum(b),1)) && max_iter < 10_000_000
    @warn "calculations not complete. Adjust the number of iterations (currently $max_iter)"
  elseif res[2] == 3
    @warn "maximal number of iterations reached ($max_iter)"
  elseif res[2] == -1
    @warn "algorithm exited with impossible status"
  end
  res[1]
end



"""
    ``emd_bonneel_with_plan(a, b, c; max_iter = 10_000_000)``

Calculate the "Earth Mover's distance" between
probability vectors `a` and `b` under cost matrix `c`.  Currently, only arrays
with data type `Float64` are supported.

# Arguments
- `a::Vector{Float64}`: Probability vector of length ``m``.
- `b::Vector{Float64}`: Probability vector of length ``n``.
- `c::Matrix{Float64}`: Cost matrix of dimension ``m x n``.
- `max_iter::Int64`: Maximal number of iterations.

# Returns
- `res` : The value of "Earth Mover's distance" between `a` and `b` under cost `c`.
- `plan::Matrix{Float64}`: The optimal coupling between `a` and `b` with cost `c`.

Throws a warning if the problem is unbounded, is infeasible, or if the maximal
number of iterations has been reached. See also [`emd_bonneel`](@ref).
"""
function emd_bonneel_with_plan(a :: Vector{Float64}, b :: Vector{Float64}, c :: Matrix{Float64}; max_iter :: Int = 10_000_000)
  if (length(a) != size(c)[1] || length(b) != size(c)[2] )
    error("The length of the weight vectors does not match the dimensions of the cost matrix")
  end
  max_iter = max(0, max_iter)
  plan = zeros(length(a), length(b))
  res = ccall( (:emd_bonneel_with_plan, lib_emd_bonneel)
             , Tuple{Float64, Int}
             , (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Int64, Csize_t)
             , a, b, c, plan, length(a), length(b), max_iter)
  if res[2] == 1
    @warn "problem unbounded"
  elseif res[2] == 2 && (!isapprox(sum(a),1) ||  !isapprox(sum(b),1))
    @warn "problem infeasible. Check that a and b are in the simplex"
  elseif res[2] == 2 && (isapprox(sum(a),1) &&  isapprox(sum(b),1)) && max_iter < 10_000_000
    @warn "calculations not complete. Adjust the number of iterations (currently $max_iter)"
  elseif res[2] == 3
    @warn "maximal number of iterations reached ($max_iter)"
  elseif res[2] == -1
    @warn "algorithm exited with impossible status"
  end
  return res[1], plan
end


# ---------------------------------------------------------------------------- #
# ThrustSort
# ---------------------------------------------------------------------------- #

# Memory control for thrustsort
"""
    ``check_GPU_mem_thrust_vec(v::CuArray)``

Checks whether there is sufficent GPU-memory for performing `thrustsort!(v)`. If not, it will free memory held by Julia.
"""
function check_GPU_mem_thrust_vec(v::CuArray)
  avm = CUDA.available_memory()
  reqm = sizeof(v)

  if (avm-1.6*reqm) < 0
    GC.gc()
    CUDA.reclaim()
    print("Reclaimed Memory.\n")
  end

end

# Memory control for sort_thrust
"""
    ``check_GPU_mem_thrust_mat(v::CuArray)``

Checks whether there is sufficent GPU-memory for performing [`sort_thrust!`](@ref) of `v`. If not, it will free memory held by Julia.
"""
function check_GPU_mem_thrust_mat(v::CuArray)
  avm = CUDA.available_memory()
  reqm = sizeof(v)

  if (avm-2.5*reqm) < 0
    GC.gc()
    CUDA.reclaim()
    print("Reclaimed Memory.\n")
  end

end

#
"""
    ``sort_thrust_buffer(n, m)``

Creates the buffer required by [`sort_thrust!`](@ref) for the columnwise sorting of an ``n x m`` matrix.
"""
function sort_thrust_buffer(n, m)
  segs = repeat(Int32.(1:m), inner = n)
  cu(reshape(segs, n, m))
end

"""
    ``sort_thrust!( matrix; [buffer])``

Columnwise sorting of a `matrix` of type `CuMatrix{Float32}`. The sorting requires a (reusable) `buffer` provided by the function
`sort_thrust_buffer(n, m)`, where `n` and `m` are the dimensions of `matrix`.
If the buffer is not provided, it is created temporarily on function invocation.
"""
function sort_thrust!( mat :: CuMatrix{Float32}
                     ; buffer :: CuMatrix{Int32} = sort_thrust_buffer(size(mat)...) )

  @assert length(mat) == length(buffer)

  # Julia does not communicate with CUDA --> We ensure manually that
  # we have sufficent GPU-memory for performing thrustsort
  check_GPU_mem_thrust_mat(mat)
  # emulate column-wise sorting by first sorting all values and then sort
  # according to column indices again
  thrustsort!(mat, buffer)

  check_GPU_mem_thrust_mat(mat)
  thrustsort!(buffer, mat, stable = true)
end

# ---------------------------------------------------------------------------- #
# FLB
# ---------------------------------------------------------------------------- #

# uniform weights
"""
    ``flb_cpu(C1, C2)``

Returns the First Lower Bound (FLB) of the Gromov–Wasserstein distance between ``(C1, p)`` and
``(C2, q)``, where ``p`` and ``q`` are assumed to be uniform measures. Called by [`flb`](@ref) if ``cuda == false``.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.

# Returns
- `res` : The value of FLB.

""" 
function flb_cpu(C1, C2)
    m = size(C1, 1)
    n = size(C2, 1)

    ec1 = sqrt.(1/m*sum(abs2, C1, dims = 2))
    sort!(ec1, dims = 1)
  
    ec2 = sqrt.(1/n*sum(abs2, C2, dims = 2))
    sort!(ec2, dims = 1)
    
    res = Wasserstein1d(ec1,ec2, 2)
end

"""
    ``flb_gpu(C1, C2)``

Returns the First Lower Bound (FLB) of the Gromov–Wasserstein distance between ``(C1, p)`` and
``(C2, q)``, where ``p`` and ``q`` are assumed to be uniform measures. Called by [`flb`](@ref) if ``cuda == true``.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.

# Returns
- `res` : The value of FLB.

"""  
function flb_gpu(C1, C2)
    m = size(C1, 1)
    n = size(C2, 1)
      
    ec1 = safe_sqrt.(1/m*(sum(abs2, C1, dims = 2)))
    # Julia does not communicate with CUDA --> We ensure manually that
    # we have sufficent GPU-memory for performing thrustsort
    check_GPU_mem_thrust_vec(ec1)
    thrustsort!(ec1)
  
    ec2 = safe_sqrt.(1/n*(sum(abs2, C2, dims = 2)))
    check_GPU_mem_thrust_vec(ec2)  
    thrustsort!(ec2)
  
    if m == n
      res = 1/m*sum((ec1-ec2).^2)
    else
      indx, indy, mass_diff = gpu_helper_function(m,n)
  
      new_ec1 = ec1[indx,:]
      new_ec2 = ec2[indy,:]
    
      res = sum((new_ec1-new_ec2).^2 .*mass_diff)
    end 
  
    return res
  
end

"""
    ``flb(C1, C2; cuda = CUDA.functional())``

Returns the First Lower Bound (FLB) of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between ``(C1, p)`` and
 ``(C2, q)``, where ``p`` and ``q`` are assumed to be uniform measures.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).

# Returns
- `res` : The value of FLB.

See also [`Euc_flb`](@ref).
"""  
function flb(C1, C2; cuda = CUDA.functional())
    if cuda
        # convert input data to gpu memory
        # makes the data Float32 for performance reasons  
        C1  = CuArray{Float32}(C1)
        C2  = CuArray{Float32}(C2)  
        res = flb_gpu(C1, C2)
    else 
        res = flb_cpu(C1, C2)
    end

    return res
end


# Euclidean

"""
    ``Euc_flb(x, y; cuda = CUDA.functional())``

Returns the First Lower Bound (FLB) of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between the uniform, Euclidean metric measure spaces 
``(C1, p)`` and ``(C2, q)``, i.e., ``C1`` and ``C2`` are
the Euclidean distance matrices of ``x`` and ``y`` and ``p`` and ``q`` are assumed to be uniform measures.

# Arguments
- `x`: Matrix of dimension ``m x r`` that encodes the locations of ``m`` points in ``ℝʳ``.
- `y`: Matrix of dimension ``n x s`` that encodes the locations of ``n`` points in ``ℝˢ``.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).

# Returns
- `res` : The value of FLB.

See also [`flb`](@ref).
"""  
function Euc_flb(x, y; cuda = CUDA.functional())
    if cuda
        # convert input data to gpu memory
        # makes the data Float32 for performance reasons  
        x = CuArray{Float32}(x)
        y = CuArray{Float32}(y)
      
        x2 = sum(abs2, x, dims = 2)
        y2 = sum(abs2, y, dims = 2)
      
        dx = safe_sqrt.(x2 .+ x2' .- 2 .* (x * x'))
        dy = safe_sqrt.(y2 .+ y2' .- 2 .* (y * y'))

        res = flb_gpu(dx, dy)
    else 
        dx = pairwise(Euclidean(), x', x')
        dy = pairwise(Euclidean(), y', y')

        res = flb_cpu(dx, dy)
    end

    return res
end


# custom weights 

"""
    ``flb(C1, C2, p::Vector, q::Vector)``

Returns the First Lower Bound (FLB) of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between ``(C1, p)`` and 
``(C2, q)``.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `p`: Probability vector of length ``m``.
- `q`: Probability vector of length ``n``.

# Returns
- `res` : The value of FLB.

For non-uniform probability vectors no GPU-support is provided. 

# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli. "The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.
"""
function flb(C1, C2, p::Vector, q::Vector)
  m = size(C1, 1)
  n = size(C2, 1)

  ec1 = sqrt.(sum( p.*C1.^2, dims = 1))
  orderec1 = sortperm(ec1[1,:])

  ec2 = sqrt.(sum(q.*C2.^2, dims = 1))
  orderec2 = sortperm(ec2[1,:])
  
  res = Wasserstein1d(ec1[orderec1], ec2[orderec2], p[orderec1], q[orderec2], 2)
end



# ---------------------------------------------------------------------------- #
# SLB
# ---------------------------------------------------------------------------- #

"""
    ``slb_cpu(C1, C2)``

Returns the Second Lower Bound (SLB) of the Gromov–Wasserstein distance between ``(C1, p)`` and
``(C2, q)``, where ``p`` and ``q`` are assumed to be uniform measures. Called by [`slb`](@ref) if ``cuda == false``.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.

# Returns
- `res` : The value of SLB.

"""  
function slb_cpu(C1, C2)
    m = size(C1, 1)
    n = size(C2, 1)
  
    C1 = reshape(C1,m^2,1)
    sort!(C1, dims = 1)
  
    C2 = reshape(C2,n^2,1)
    sort!(C2, dims = 1)
  
    res = Wasserstein1d(C1,C2, 2)
  
  end
  
"""
    ``slb_gpu(C1, C2)``

Returns the Second Lower Bound (SLB) of the Gromov–Wasserstein distance between ``(C1, p)`` and
``(C2, q)``, where ``p`` and ``q`` are assumed to be uniform measures. Called by [`slb`](@ref) if ``cuda == true``.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.

# Returns
- `res` : The value of SLB.

"""  
  function slb_gpu(C1, C2)
    m = size(C1, 1)
    n = size(C2, 1)
  
  
    C1 = reshape(C1, m^2, 1)
    C2 = reshape(C2, n^2, 1)
    
    # Julia does not communicate with CUDA --> We ensure manually that
    # we have sufficent GPU-memory for performing thrustsort
    check_GPU_mem_thrust_vec(C1)
    thrustsort!(C1)
  
    check_GPU_mem_thrust_vec(C2)
    thrustsort!(C2)
  
    if m == n
       res = 1/(m)^2*sum((C1-C2).^2)
    else
      ind1, ind2, mass_diff = gpu_helper_function(m^2,n^2) # runtime for n neq m is improvable!
  
      C1 = C1[ind1,:]
      C2 = C2[ind2,:]
    
      res = sum((C1-C2).^2 .*mass_diff)
    end
  
    return res
  end
  
"""
    ``slb(C1, C2; cuda = CUDA.functional())``

Returns the Second Lower Bound (SLB) of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between ``(C1, p)`` and
 ``(C2, q)``, where ``p`` and ``q`` are assumed to be uniform measures.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).

# Returns
- `res` : The value of SLB.

See also [`Euc_slb`](@ref).

"""  
  function slb(C1, C2; cuda = CUDA.functional(), kwargs...)  
    
    if cuda
        # convert input data to gpu memory
        # makes the data Float32 for performance reasons  
        C1  = CuArray{Float32}(C1)
        C2  = CuArray{Float32}(C2)  
        res = slb_gpu(C1, C2)
    else
        # slb_cpu changes the matrices C1 & C2 --> copies are passed

        res = slb_cpu(copy(C1), copy(C2))
    end
    
    return res
  end


# Euclidean
"""
    ``Euc_slb(x, y; cuda = CUDA.functional())``

Returns the Second Lower Bound (SLB) of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between the uniform, Euclidean metric measure spaces 
``(C1, p)`` and ``(C2, q)``, i.e., ``C1`` and ``C2`` are
the Euclidean distance matrices of ``x`` and ``y`` and ``p`` and ``q`` are assumed to be uniform measures.

# Arguments
- `x`: Matrix of dimension ``m x r`` that encodes the locations of ``m`` points in ``ℝʳ``.
- `y`: Matrix of dimension ``n x s`` that encodes the locations of ``n`` points in ``ℝˢ``.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).

# Returns
- `res` : The value of SLB.

See also [`slb`](@ref).

# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli. "The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.
"""  
function Euc_slb(x, y; cuda = CUDA.functional())
    if cuda
        # convert input data to gpu memory
        # makes the data Float32 for performance reasons  
        x = CuArray{Float32}(x)
        y = CuArray{Float32}(y)
      
        x2 = sum(abs2, x, dims = 2)
        y2 = sum(abs2, y, dims = 2)
      
        dx = safe_sqrt.(x2 .+ x2' .- 2 .* (x * x'))
        dy = safe_sqrt.(y2 .+ y2' .- 2 .* (y * y'))

        res = slb_gpu(dx, dy)
    else 
        dx = pairwise(Euclidean(), x', x')
        dy = pairwise(Euclidean(), y', y')

        res = slb_cpu(dx, dy)
    end

    return res
end

# Custom weights
"""
    ``slb(C1, C2, p::Vector, q::Vector)``

Returns the Second Lower Bound (SLB) of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between ``(C1, p)`` and 
``(C2, q)``.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `p`: Probability vector of length ``m``.
- `q`: Probability vector of length ``n``.

# Returns
- `res` : The value of SLB.

For non-uniform probability vectors no GPU-support is provided.

# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli. "The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.
"""
function slb(C1, C2, p, q)
  m = size(C1, 1)
  n = size(C2, 1)

  C1 = vec(C1)
  p = vec(p*p')
  ind1 = sortperm(C1)

  C2 = vec(C2)
  q = vec(q*q')
  ind2 = sortperm(C2)

  res = Wasserstein1d(C1[ind1], C2[ind2], p[ind1], q[ind2], 2)

end


# ---------------------------------------------------------------------------- #
# TLB
# ---------------------------------------------------------------------------- #

"""
    ``tlb_cpu(C1, C2; plan = false, max_iter = 10_000_000, kwargs...)``
    
Returns the Third Lower Bound (TLB) of the Gromov–Wasserstein distance between ``(C1, p)`` and
 ``(C2, q)``, where ``p`` and ``q`` are assumed to be uniform measures. Called by [`tlb`](@ref) if ``cuda == false``.

 # Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `plan`: Boolean that signals if an optimal coupling should be returned.
- `max_iter::Int64`: Maximal number of iterations for calculating the Earth Mover's distance.

# Returns
- `res` : The value of TLB.
- `plan` : Optimal coupling between ``(C1, p)`` and ``(C2, q)`` (only returned if ``plan == true``) 
 """
function tlb_cpu(C1, C2; plan = false, max_iter = 10_000_000,  kwargs...)
  
  m = size(C1, 1)
  n = size(C2, 1)
  
  # For consistency, let BLAS use the same number of threads as there are active
  # julia threads
    
  torig = BLAS.get_num_threads()
  t = Threads.nthreads()
  BLAS.set_num_threads(t)
  
    
  Threads.@threads for j in 1:m
    sort!(@view(C1[:, j]))
  end
  
  Threads.@threads for j in 1:n
    sort!(@view(C2[:, j]))
  end
    
  if m == n
      dC1 = sum(abs2, C1, dims=1)
      dC2 = sum(abs2, C2, dims=1)
  
      c = (- 2 .* (C1' * C2).+ dC1'.+ dC2) ./ n
  else
      indx, indy, mass_diff = cpu_helper_function(m,n)

      new_C1 = C1[indx,:]
      new_C2 = C2[indy,:]

      new_C1 = new_C1.*sqrt.(mass_diff)
      new_C2 = new_C2.*sqrt.(mass_diff)

      C1_sum = reshape(sum(abs2, new_C1, dims=1),m,1)
      C2_sum = sum(abs2, new_C2, dims=1)

      c = (- (2.0) .* (new_C1' * new_C2).+ C1_sum.+ C2_sum)
  end

  BLAS.set_num_threads(torig)

  c = Array{Float64}(c)
  a = fill(1/m, m)
  b = fill(1/n, n)
  if plan
    res = emd_bonneel_with_plan(a, b, c; max_iter = max_iter)
  else 
    res = emd_bonneel(a, b, c; max_iter = max_iter)
  end
  return(res)
end

"""
    ``tlb_gpu(C1, C2; plan = false, max_iter = 10_000_000, buffer1 = nothing, buffer2 = nothing, kwargs...)``
    
Returns the Third Lower Bound (TLB) of the Gromov–Wasserstein distance between ``(C1, p)`` and
 ``(C2, q)``, where ``p`` and ``q`` are assumed to be uniform measures. Called by [`tlb`](@ref) if ``cuda == true``.

 # Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `plan`: Boolean that signals if an optimal coupling should be returned.
- `max_iter::Int64`: Maximal number of iterations for calculating the Earth Mover's distance.
- `buffer1`: Thrustsort-Buffer provided by the function `sort_thrust_buffer(m, m)`.
- `buffer2`: Thrustsort-Buffer provided by the function `sort_thrust_buffer(n, n)`.

# Returns
- `res` : The value of TLB.
- `plan` : Optimal coupling between ``(C1, p)`` and ``(C2, q)`` (only returned if ``plan == true``). 
 """
function tlb_gpu(C1, C2; plan = false, max_iter = 10_000_000, buffer1 = nothing, buffer2 = nothing, kwargs...)
  m = size(C1, 1)
  n = size(C2, 1)

  # Buffers can be pre-allocated to speed up the sorting
  # Useful for Bootstrap application
  if isnothing(buffer1)
    sort_thrust!(C1)
  else
    sort_thrust!(C1; buffer = buffer1)
  end

  if isnothing(buffer2)
    if m == n && (!isnothing(buffer1))
      sort_thrust!(C2; buffer = buffer1)
    else
      sort_thrust!(C2)
    end
  else
    sort_thrust!(C2, buffer = buffer2)
  end

  if n == m
    dC1 = sum(abs2, C1, dims=1)
    dC2 = sum(abs2, C2, dims=1)

    c = (- 2 .* (C1' * C2).+ dC1'.+ dC2) ./ n
  else
    indx, indy, mass_diff = gpu_helper_function(m,n)

    new_C1 = C1[indx,:]
    new_C2 = C2[indy,:]

    new_C1 = new_C1.*sqrt.(mass_diff)
    new_C2 = new_C2.*sqrt.(mass_diff)

    C1_sum = reshape(sum(abs2, new_C1, dims=1),m,1)
    C2_sum = sum(abs2, new_C2, dims=1)

    c = (- (2.0) .* (new_C1' * new_C2).+ C1_sum.+ C2_sum)

  end  
  # convert cost matrix back to cpu memory
  c = Array{Float64}(c)
  a = fill(1/m, m)
  b = fill(1/n, n)

  if plan
    res = emd_bonneel_with_plan(a, b, c, max_iter = max_iter)
  else 
    res = emd_bonneel(a, b, c; max_iter = max_iter)
  end
  return res
end
  
"""
    ``tlb(C1, C2; plan = false, cuda = CUDA.functional(), max_iter = 10_000_000, kwargs...)``
    
Returns the Third Lower Bound (TLB) of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between ``(C1, p)`` and
 ``(C2, q)``, where ``p`` and ``q`` are assumed to be uniform measures.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `plan`: Boolean that signals if an optimal coupling should be returned.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `max_iter::Int64`: Maximal number of iterations for calculating the Earth Mover's distance.
- `kwargs`: Additional arguments for the function [`MMSpaces.tlb_gpu`](@ref).

# Returns
- `res` : The value of TLB.
- `plan` : Optimal coupling between ``(C1, p)`` and ``(C2, q)`` (only returned if ```plan == true``). 

See also [`Euc_tlb`](@ref).
"""  
function tlb(C1, C2; plan = false, cuda = CUDA.functional(), max_iter = 10_000_000, kwargs...)
    
  if cuda
      # convert input data to gpu memory
      # makes the data Float32 for performance reasons  
      C3  = CUDA.zeros(size(C1)...)
      C3  = copyto!(C3, C1) 
      C4  = CUDA.zeros(size(C2)...)
      C4  = copyto!(C4, C2) 
      res = tlb_gpu(C3, C4; plan = plan, max_iter = max_iter, kwargs...)
  else
      # tlb_cpu changes the matrices C1 & C2 --> copies are passed
      res = tlb_cpu(copy(C1), copy(C2); plan = plan, max_iter = max_iter, kwargs...)
  end

  return res
end


# Euclidean
"""
    ``Euc_tlb(x, y; plan = false, cuda = CUDA.functional(), max_iter = 10_000_000, kwargs...)``

Returns the Thrid Lower Bound (TLB) of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between the uniform, Euclidean metric measure spaces 
``(C1, p)`` and ``(C2, q)``, i.e., ``C1`` and ``C2`` are
the Euclidean distance matrices of ``x`` and ``y`` and ``p`` and ``q`` are assumed to be uniform measures.

# Arguments
- `x`: Matrix of dimension ``m x r`` that encodes the locations of ``m`` points in ``ℝʳ``.
- `y`: Matrix of dimension ``n x s`` that encodes the locations of ``n`` points in ``ℝˢ``.
- `plan`: Boolean that signals if an optimal coupling should be returned.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `max_iter::Int64`: Maximal number of iterations for calculating the Earth Mover's distance.
- `kwargs`: Additional arguments for the function [`MMSpaces.tlb_gpu`](@ref).

# Returns
- `res` : The value of TLB.
- `plan` : Optimal coupling between ``(C1, p)`` and ``(C2, q)`` (only returned if ```plan == true``). 

See also [`tlb`](@ref).

# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli. "The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.
"""  
function Euc_tlb(x, y; plan = false, cuda = CUDA.functional(), max_iter = 10_000_000, kwargs...)
  if cuda
      # convert input data to gpu memory
      # makes the data Float32 for performance reasons  
      x = CuArray{Float32}(x)
      y = CuArray{Float32}(y)
    
      x2 = sum(abs2, x, dims = 2)
      y2 = sum(abs2, y, dims = 2)
    
      dx = safe_sqrt.(x2 .+ x2' .- 2 .* (x * x'))
      dy = safe_sqrt.(y2 .+ y2' .- 2 .* (y * y'))

      res = tlb_gpu(dx, dy; plan = plan, max_iter = max_iter, kwargs...)
  else 
      dx = pairwise(Euclidean(), x', x')
      dy = pairwise(Euclidean(), y', y')

      res = tlb_cpu(dx, dy; plan = plan, max_iter = max_iter, kwargs...)
  end

  return res
end

# custom weights
"""
    ``tlb(C1, C2, p, q; plan = false, max_iter = 10_000_000)``

Returns the Third Lower Bound (TLB) of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between ``(C1, p)`` and 
``(C2, q)``.

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `p`: Probability vector of length ``m``.
- `q`: Probability vector of length ``n``.
- `plan`: Boolean that signals if an optimal coupling should be returned.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `max_iter::Int64`: Maximal number of iterations for calculating the Earth Mover's distance.

# Returns
- `res` : The value of TLB.
- `plan` : Optimal coupling between ``(C1, p)`` and ``(C2, q)`` (only returned if ```plan == true``). 

For non-uniform probability vectors no GPU-support is provided. 

# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli. "The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.
"""
function tlb(C1, C2, p, q; plan = false, max_iter = 10_000_000)

  m = size(C1, 1)
  n = size(C2, 1)

  # For consistency, let BLAS use the same number of threads as there are active
  # julia threads
  torig = BLAS.get_num_threads()
  t = Threads.nthreads()
  BLAS.set_num_threads(t)

  indices1 =zeros(Int32, m, m)
  Threads.@threads for j in 1:m
    indices1[:,j] = sortperm(@view(C1[:, j]))
  end

  indices2 = zeros(Int32, n, n)
  Threads.@threads for j in 1:n
    indices2[:,j] = sortperm(@view(C2[:, j]))
  end

  c = pairwise((x,y) -> Wasserstein1d(C1[x,indices1[:,x]], C2[y,indices2[:,y]], p[indices1[:,x]], q[indices2[:,y]], 2) , 1:m, 1:n)
  c = Array{Float64}(c)

  BLAS.set_num_threads(torig)

  if plan
    res = emd_bonneel_with_plan(a,b,c)
  else 
    res = emd_bonneel(a, b, c; max_iter = max_iter)
  end
    return(res)
end

# ---------------------------------------------------------------------------- #
# FLB-Bootstrap
# ---------------------------------------------------------------------------- #
"""
    ``FLB_Bootstrap(C1, C2, m1, m2; number = 1000, cuda = CUDA.functional(), flb_value = nothing)``

Provides the Bootrap sample required for performing [`FLB_Test`](@ref) for general cost matrices.

# Arguments
- `C1`: A symmetric cost matrix of dimension ``m x m``.
- `C2`: A symmetric cost matrix of dimension ``n x n``. 
- `m1`: Number of resamples from ``C1`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `m2`: Number of resamples from ``C2`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `number`: Size of the Bootstrap sample.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `flb_value`: In order to avoid calculating it multiple times, the value of FLB required for the Bootstrap sample can be passed here (optional). 

# Returns
- `result` : A Bootstrap sample required for performing [`FLB_Test`](@ref).

# References

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.
""" 
function FLB_Bootstrap(C1, C2, m1, m2; number = 1000, cuda = CUDA.functional(), flb_value = nothing)
  m = size(C1, 1)
  n = size(C2, 1)
  
  if isnothing(flb_value)
    flb_value = flb(C1, C2, cuda = cuda)
  end
  result = zeros(number)


 for  i in 1:number

    sampled_numbers_C1 = rand(1:m, m1)
    sampled_numbers_C2 = rand(1:n, m2)
        
    bootsamp_C1 = C1[sampled_numbers_C1, sampled_numbers_C1]
    bootsamp_C2 = C2[sampled_numbers_C2, sampled_numbers_C2]
     
    result[i] = sqrt(m1*m2/(m1 + m2))*(flb(bootsamp_C1, bootsamp_C2, cuda = cuda) - flb_value)
 end

 return result

end

"""
    ``Euc_FLB_Bootstrap(x, y, mx, my; number = 1000, cuda = CUDA.functional(), flb_value = nothing)``

Provides the Bootrap sample required for performing [`FLB_Test`](@ref) based on samples from Euclidean spaces.

# Arguments
- `x`: Matrix of dimension ``m x r`` that encodes the locations of ``m`` points in ``ℝʳ``.
- `y`: Matrix of dimension ``n x s`` that encodes the locations of ``n`` points in ``ℝˢ``.
- `m1`: Number of resamples from ``C1`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `m2`: Number of resamples from ``C2`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `number`: Size of the Bootstrap sample.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `flb_value`: In order to avoid calculating it multiple times, the value of FLB required for the Bootstrap sample can be passed here (optional). 

# Returns
- `result` : A Bootstrap sample required for performing [`FLB_Test`](@ref).

# References

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.
""" 
 function Euc_FLB_Bootstrap(x, y, mx, my; number = 1000, cuda = CUDA.functional(), flb_value = nothing)
  m = size(x, 1)
  n = size(y, 1)
  
  if isnothing(flb_value)
    flb_value = Euc_flb(x, y, cuda = cuda)
  end
  result = zeros(number)


 for  i in 1:number

    sampled_numbers_x = rand(1:m, mx)
    sampled_numbers_y = rand(1:n, my)
    
    bootsamp_x = x[sampled_numbers_x, :]
    bootsamp_y = y[sampled_numbers_y, :]
    
    result[i] = sqrt(mx*my/(mx + my))*(Euc_flb(bootsamp_x, bootsamp_y, cuda = cuda) - flb_value)
 end

 return result

 end

# ---------------------------------------------------------------------------- #
# SLB-Bootstrap
# ---------------------------------------------------------------------------- #
"""
    ``SLB_Bootstrap(C1, C2, m1, m2; number = 1000, cuda = CUDA.functional(), slb_value = nothing)``

Provides the Bootrap sample required for performing [`SLB_Test`](@ref) for general cost matrices.

# Arguments
- `C1`: A symmetric cost matrix of dimension ``m x m``.
- `C2`: A symmetric cost matrix of dimension ``n x n``. 
- `m1`: Number of resamples from ``C1`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `m2`: Number of resamples from ``C2`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `number`: Size of the Bootstrap sample.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `slb_value`: In order to avoid calculating it multiple times, the value of SLB required for the Bootstrap sample can be passed here (optional). 

# Returns
- `result` : A Bootstrap sample required for performing [`SLB_Test`](@ref).

# References

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.
""" 
function SLB_Bootstrap(C1, C2, m1, m2; number = 1000, cuda = CUDA.functional(), slb_value = nothing)
  m = size(C1, 1)
  n = size(C2, 1)
  
  if isnothing(slb_value)
    slb_value = slb(C1, C2, cuda = cuda)
  end
  result = zeros(number)


 for  i in 1:number

    sampled_numbers_C1 = rand(1:m, m1)
    sampled_numbers_C2 = rand(1:n, m2)
        
    bootsamp_C1 = C1[sampled_numbers_C1, sampled_numbers_C1]
    bootsamp_C2 = C2[sampled_numbers_C2, sampled_numbers_C2]
     
    result[i] = sqrt(m1*m2/(m1 + m2))*(slb(bootsamp_C1, bootsamp_C2, cuda = cuda) - slb_value)
 end

 return result

end

"""
    ``Euc_SLB_Bootstrap(x, y, mx, my; number = 1000, cuda = CUDA.functional(), slb_value = nothing)``

Provides the Bootrap sample required for performing [`SLB_Test`](@ref) based on samples Euclidean spaces.

# Arguments
- `x`: Matrix of dimension ``m x r`` that encodes the locations of ``m`` points in ``ℝʳ``.
- `y`: Matrix of dimension ``n x s`` that encodes the locations of ``n`` points in ``ℝˢ``.
- `m1`: Number of resamples from ``C1`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `m2`: Number of resamples from ``C2`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `number`: Size of the Bootstrap sample.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `slb_value`: In order to avoid calculating it multiple times, the value of SLB required for the Bootstrap sample can be passed here (optional). 

# Returns
- `result` : A Bootstrap sample required for performing [`SLB_Test`](@ref).

# References

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.
""" 
 function Euc_SLB_Bootstrap(x, y, mx, my; number = 1000, cuda = CUDA.functional(), slb_value = nothing)
  m = size(x, 1)
  n = size(y, 1)
  
  if isnothing(slb_value)
    slb_value = Euc_slb(x, y, cuda = cuda)
  end

  result = zeros(number)


  for  i in 1:number

    sampled_numbers_x = rand(1:m, mx)
    sampled_numbers_y = rand(1:n, my)
    
    
    bootsamp_x = x[sampled_numbers_x, :]
    bootsamp_y = y[sampled_numbers_y, :]
    

 
    result[i] = sqrt(mx*my/(mx + my))*(Euc_slb(bootsamp_x, bootsamp_y, cuda = cuda) - slb_value)
 end

 return result

end


# ---------------------------------------------------------------------------- #
# TLB-Bootstrap
# ---------------------------------------------------------------------------- #
"""
    ``TLB_Bootstrap(C1, C2, m1, m2; number = 1000, cuda = CUDA.functional(), tlb_value = nothing, kwargs...)``

Provides the Bootrap sample required for performing [`TLB_Test`](@ref) for general cost matrices.

# Arguments
- `C1`: A symmetric cost matrix of dimension ``m x m``.
- `C2`: A symmetric cost matrix of dimension ``n x n``. 
- `m1`: Number of resamples from ``C1`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `m2`: Number of resamples from ``C2`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `number`: Size of the Bootstrap sample.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `tlb_value`: In order to avoid calculating it multiple times, the value of TLB required for the Bootstrap sample can be passed here (optional). 
- `kwargs...`: May contain additional parameters for [`emd_bonneel`](@ref).

# Returns
- `result` : A Bootstrap sample required for performing [`TLB_Test`](@ref).

# References

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.
""" 
function TLB_Bootstrap(C1, C2, m1, m2; number = 1000, cuda = CUDA.functional(), tlb_value = nothing, kwargs...)
  m = size(C1, 1)
  n = size(C2, 1)
  

  if isnothing(tlb_value)
    tlb_value = tlb(C1, C2, cuda = cuda; kwargs...)
  end

  buffer1 = nothing
  buffer2 = nothing

  if cuda
    if m1 == m2
      buffer1 = sort_thrust_buffer(m1, m1)
    else
      buffer1 = sort_thrust_buffer(m1, m1)
      buffer2 = sort_thrust_buffer(m2, m2)
    end
  end

  result = zeros(number)


 for  i in 1:number

    sampled_numbers_C1 = rand(1:m, m1)
    sampled_numbers_C2 = rand(1:n, m2)
        
    bootsamp_C1 = C1[sampled_numbers_C1, sampled_numbers_C1]
    bootsamp_C2 = C2[sampled_numbers_C2, sampled_numbers_C2]
     
    result[i] = sqrt(m1*m2/(m1 + m2))*(tlb(bootsamp_C1, bootsamp_C2, cuda = cuda; buffer1 = buffer1, buffer2 = buffer2, kwargs...) - tlb_value)
 end

 return result

end

"""
    ``Euc_TLB_Bootstrap(x, y, mx, my; number = 1000, cuda = CUDA.functional(), tlb_value = nothing, kwargs...)``

Provides the Bootrap sample required for performing [`TLB_Test`](@ref) based on samples from Euclidean spaces.

# Arguments
- `x`: Matrix of dimension ``m x r`` that encodes the locations of ``m`` points in ``ℝʳ``.
- `y`: Matrix of dimension ``n x s`` that encodes the locations of ``n`` points in ``ℝˢ``.
- `m1`: Number of resamples from ``C1`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `m2`: Number of resamples from ``C2`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `number`: Size of the Bootstrap sample.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `tlb_value`: In order to avoid calculating it multiple times, the value of TLB required for the Bootstrap sample can be passed here (optional). 
- `kwargs...`: May contain additional parameters for [`emd_bonneel`](@ref).

# Returns
- `result` : A Bootstrap sample required for performing [`TLB_Test`](@ref).

# References

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.
""" 
 function Euc_TLB_Bootstrap(x, y, mx, my; number = 1000, cuda = CUDA.functional(), tlb_value = nothing, kwargs...)
  m = size(x, 1)
  n = size(y, 1)
  

  if isnothing(tlb_value)
    tlb_value = Euc_tlb(x,y, cuda = cuda; kwargs...)
  end


  buffer1 = nothing
  buffer2 = nothing

  if cuda
    if mx == my
      buffer1 = sort_thrust_buffer(mx, mx)
    else
      buffer1 = sort_thrust_buffer(mx, mx)
      buffer2 = sort_thrust_buffer(my, my)
    end
  end

  result = zeros(number)


 for  i in 1:number

    sampled_numbers_x = rand(1:m, mx)
    sampled_numbers_y = rand(1:n, my)
    
    
    bootsamp_x = x[sampled_numbers_x, :]
    bootsamp_y = y[sampled_numbers_y, :]
    

 
    result[i] = sqrt(mx*my/(mx + my))*(Euc_tlb(bootsamp_x, bootsamp_y, cuda = cuda;  buffer1 = buffer1, buffer2 = buffer2, kwargs...) - tlb_value)
 end

 return result

 end

# ---------------------------------------------------------------------------- #
# FLB-Test
# ---------------------------------------------------------------------------- #
"""
    ``FLB_Test(C1, C2, delta, m1,  m2; hypo ="equiv", number = 1000, cuda = CUDA.functional(), boot_samp_ret = false, force_euc = false)``

Equivalence testing or testing for relevant distances based on the First Lower Bound (FLB) of the Gromov-Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)).
More precisely, given cost matrices ``C1`` and ``C2`` (resp. Euclidean coordinates out of which ``C1`` and ``C2``
can be determined) of two independent samples of ``(C₁, p)`` and ``(C₂, q)`` an asymptotic level ``α`` test for

``H₀ = FLB((C₁,p), (C₂, q)) > Δ`` versus ``H₁ = FLB((C₁, p), (C₂, q)) ≤ Δ``  `(Equivalence Testing)`

or 

``H₀ = FLB((C₁,p), (C₂, q)) ≤ Δ`` versus ``H₁ = FLB((C₁, p), (C₂, q)) > Δ``  `(Testing for Relevant Differences)`

is performed.

# Arguments
- `C1`: Either a symmetric cost matrix of dimension ``m x m`` or an ``m x r`` matrix that encodes the locations of ``m`` points in ``ℝʳ``.
- `C2`: Either a symmetric cost matrix of dimension ``n x n`` or an ``n x s`` matrix that encodes the locations of ``n`` points in ``ℝˢ``. Note that if C1 encodes a cost matrix/Euclidean coordinates, C2 is assumed to do the same.
- `delta`: Parameter (Float) for the formulation of the hypothesis.
- `m1`: Number of resamples from ``C1`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `m2`: Number of resamples from ``C2`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
- `hypo`: A string that determines the testing problem. Either ``equiv`` for equivalence testing or ``rel_diff`` for testing for relevant differences.
- `number`: Size of the Bootstrap sample for estimating the required quantiles.
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `boot_samp_ret`: Boolean that signals wheter the Bootstrap sample should be returned (if ``true``, it will be returned).
- `force_euc`: Boolean that signals that the  square matrices ``C1`` and ``C2`` should be interpreted as Euclidean coordinates (if ``true``, they will be interpreted as Euclidean coordinates).

# Returns
- `flb_value` : The value of FLB.
- `pval` : The p-value for the testing problem considered.
- `boot_samp` : The bootstrap sample used for the estimation of the quantiles required for testing (only returned if ``boot_samp_ret == true``).

See also [`SLB_Test`](@ref) and [`TLB_Test`](@ref). For the theoretical foundations of this test see Mordant et al. (2023).

# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli. "The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.
""" 
 function FLB_Test(C1, C2, delta, m1,  m2; hypo ="equiv", number = 1000, cuda = CUDA.functional(), boot_samp_ret = false, force_euc = false)
  m,k = size(C1)
  n,l = size(C2)

  if m == k && n == l && force_euc == false
    lb = flb
    bootstrap = FLB_Bootstrap
  else
    lb = Euc_flb
    bootstrap = Euc_FLB_Bootstrap
  end

  flb_value = lb(C1, C2, cuda = cuda)
  boot_samp = bootstrap(C1,  C2, m1,  m2, number = number, cuda = cuda, flb_value = flb_value)

  if hypo == "equiv"
    pval = ecdf(boot_samp)(sqrt(n*m/(n+m))*(flb_value-delta))
  elseif hypo == "rel_diff"
    pval = 1 - ecdf(boot_samp)(sqrt(n*m/(n+m))*(flb_value-delta))
  else
    error("Invalid choice for the variable hypo.")
  end

  if boot_samp_ret
    res = (flb_value, pval, boot_samp)
  else
    res = (flb_value, pval)
  end

end 


# ---------------------------------------------------------------------------- #
# SLB-Test
# ---------------------------------------------------------------------------- #
"""
    ``SLB_Test(C1, C2, delta, m1,  m2; hypo ="equiv", number = 1000, cuda = CUDA.functional(), boot_samp_ret = false, force_euc = false)``

  Equivalence testing or testing for relevant distances based on the Second Lower Bound (SLB) of the Gromov-Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)).
  More precisely, given cost matrices ``C1`` and ``C2`` (resp.
  Euclidean coordinates out of which ``C1`` and ``C2`` can be determined) of two independent samples
  of ``(C₁, p)`` and ``(C₂, q)`` an asymptotic level ``α`` test for
  
  ``H₀ = SLB((C₁,p), (C₂, q)) > Δ`` versus ``H₁ = SLB((C₁, p), (C₂, q)) ≤ Δ``  `(Equivalence Testing)`
  
  or 
  
  ``H₀ = SLB((C₁,p), (C₂, q)) ≤ Δ`` versus ``H₁ = SLB((C₁, p), (C₂, q)) > Δ``  `(Testing for Relevant Differences)`
  
  is performed.
  
  # Arguments
  - `C1`: Either a symmetric cost matrix of dimension ``m x m`` or an ``m x r`` matrix that encodes the locations of ``m`` points in ``ℝʳ``.
  - `C2`: Either a symmetric cost matrix of dimension ``n x n`` or an ``n x s`` matrix that encodes the locations of ``n`` points in ``ℝˢ``. Note that if C1 encodes a cost matrix/Euclidean coordinates, C2 is assumed to do the same.
  - `delta`: Parameter (Float) for the formulation of the hypothesis.
  - `m1`: Number of resamples from ``C1`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
  - `m2`: Number of resamples from ``C2`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
  - `hypo`: A string that determines the testing problem. Either ``equiv`` for equivalence testing or ``rel_diff`` for testing for relevant differences.
  - `number`: Size of the Bootstrap sample for estimating the required quantiles.
  - `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `boot_samp_ret`: Boolean that signals wheter the Bootstrap sample should be returned (if ``true``, it will be returned).
- `force_euc`: Boolean that signals that the  square matrices C1 and C2 should be interpreted as Euclidean coordinates (if ``true``, they will be interpreted as Euclidean coordinates).

# Returns
- `slb_value` : The value of the SLB.
- `pval` : The p-value for the testing problem considered.
- `boot_samp` : The bootstrap sample used for the estimation of the quantiles required for testing (only returned if `boot_samp_ret == true`).

See also [`FLB_Test`](@ref) and [`TLB_Test`](@ref). For the theoretical foundations of this test see Mordant et al. (2023).

# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli. "The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.
""" 
function SLB_Test(C1, C2, delta, m1,  m2; hypo ="equiv", number = 1000, cuda = CUDA.functional(), boot_samp_ret = false, force_euc = false)
  m,k = size(C1)
  n,l = size(C2)

  if m == k && n == l && force_euc == false
    lb = slb
    bootstrap = SLB_Bootstrap
  else
    lb = Euc_slb
    bootstrap = Euc_SLB_Bootstrap
  end

  slb_value = lb(C1, C2, cuda = cuda)
  boot_samp = bootstrap(C1, C2, m1, m2, number = number, cuda = cuda, slb_value = slb_value)

  if hypo == "equiv"
    pval = ecdf(boot_samp)(sqrt(n*m/(n+m))*(slb_value-delta))
  elseif hypo == "rel_diff"
    pval = 1 - ecdf(boot_samp)(sqrt(n*m/(n+m))*(slb_value-delta))
  else
    error("Invalid choice for the variable hypo.")
  end

  if boot_samp_ret
    res = (slb_value, pval, boot_samp)
  else
    res = (slb_value, pval)
  end

end 

# ---------------------------------------------------------------------------- #
# TLB-Test
# ---------------------------------------------------------------------------- #

"""
    ``TLB_Test(C1, C2, delta, m1,  m2; hypo ="equiv", number = 1000, cuda = CUDA.functional(), boot_samp_ret = false, force_euc = false, kwargs...)``

  Equivalence testing or testing for relevant distances based on the Third Lower Bound (TLB) of the Gromov-Wasserstein distance.
  More precisely, given cost matrices ``C1`` and ``C2`` (resp. Euclidean coordinates out of which ``C1`` and ``C2`` can be determined) 
  of two independent samples of ``(C₁, p)`` and ``(C₂, q)`` an asymptotic level ``α`` test for
  
  ``H₀ = TLB((C₁,p), (C₂, q)) > Δ`` versus ``H₁ = TLB((C₁, p), (C₂, q)) ≤ Δ``  `(Equivalence Testing)`
  
  or 
  
  ``H₀ = TLB((C₁,p), (C₂, q)) ≤ Δ`` versus ``H₁ = TLB((C₁, p), (C₂, q)) > Δ``  `(Testing for Relevant Differences)`
  
  is performed.
  
  # Arguments
  - `C1`: Either a symmetric cost matrix of dimension ``m x m`` or an ``m x r`` matrix that encodes the locations of ``m`` points in ``ℝʳ``.
  - `C2`: Either a symmetric cost matrix of dimension ``n x n`` or an ``n x s`` matrix that encodes the locations of ``n`` points in ``ℝˢ``. Note that if C1 encodes a cost matrix/Euclidean coordinates, C2 is assumed to do the same.
  - `delta`: Parameter (Float) for the formulation of the hypothesis.
  - `m1`: Number of resamples from ``C1`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
  - `m2`: Number of resamples from ``C2`` for the Bootstrap procedure on which the test is based (see Mordant et al. (2023) for more information).
  - `hypo`: A string that determines the testing problem. Either ``equiv`` for equivalence testing or ``rel_diff`` for testing for relevant differences.
  - `number`: Size of the Bootstrap sample for estimating the required quantiles.
  - `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `boot_samp_ret`: Boolean that signals wheter the Bootstrap sample should be returned (if ``true``, it will be returned).
- `force_euc`: Boolean that signals that the  square matrices C1 and C2 should be interpreted as Euclidean coordinates (if ``true``, they will be interpreted as Euclidean coordinates).
- `kwargs...`: May contain additional parameters for [`emd_bonneel`](@ref).

# Returns
- `tlb_value` : The value of TLB.
- `pval` : The p-value for the testing problem considered.
- `boot_samp` : The bootstrap sample used for the estimation of the quantiles required for testing (only returned if `boot_samp_ret == true`).

See also [`FLB_Test`](@ref) and [`SLB_Test`](@ref). For the theoretical Foundations of this test see Mordant et al. (2023).

# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli. "The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.

G. Mordant, S. Hundrieser, C. Weitkamp and A. Munk. "Robust statistical analysis of metric measure spaces". In preparation, 2023.
""" 
function TLB_Test(C1, C2, delta, m1,  m2; hypo ="equiv", number = 1000, cuda = CUDA.functional(), boot_samp_ret = false, force_euc = false, kwargs...)
  m,k = size(C1)
  n,l = size(C2)

  if m == k && n == l && force_euc == false
    lb = tlb
    bootstrap = TLB_Bootstrap
  else
    lb = Euc_tlb
    bootstrap = Euc_TLB_Bootstrap
  end

  tlb_value = lb(C1, C2; kwargs ...)
  boot_samp = bootstrap(C1,  C2, m1,  m2, number = number, tlb_value = tlb_value, cuda = cuda; kwargs...)
  
  if hypo == "equiv"
    pval = ecdf(boot_samp)(sqrt(n*m/(n+m))*(tlb_value-delta))
  elseif hypo == "rel_diff"
    pval = 1 - ecdf(boot_samp)(sqrt(n*m/(n+m))*(tlb_value-delta))
  else
    error("Invalid choice for the variable hypo.")
  end

  if boot_samp_ret
    res = (tlb_value, pval, boot_samp)
  else
    res = (tlb_value, pval)
  end

  return res 
end 

