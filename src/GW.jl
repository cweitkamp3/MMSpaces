using Dates
"""
    ``tensor_product(constC, hC1, hC2, T)``
Return the tensor for the fast computation of the Gromov–Wasserstein loss of a given coupling ``T``. 
The tensor is computed as described in Proposition 1 Eq. (6) of Peyré et al. (2016).

# References
Gabriel Peyré, Marco Cuturi, and Justin Solomon, "Gromov-Wasserstein averaging of kernel and distance matrices."
International Conference on Machine Learning (ICML). 2016.
"""
function tensor_product(constC, hC1, hC2, T)
    A = - hC1*T*(hC2')
    tens = constC + A
    # tens -= tens.min()
    return tens
end

"""
    ``gw_loss(constC, hC1, hC2, T)``
Fast calculation of  the Gromov–Wasserstein loss of a given coupling ``T`` as described in Proposition 1 Eq. (6) of Peyré et al. (2016).

# References

G. Peyré, M. Cuturi, and J. Solomon, "Gromov-Wasserstein averaging of kernel and distance matrices."
International Conference on Machine Learning (ICML). 2016.
"""
function gw_loss(constC, hC1, hC2, T)
    loss = tensor_product(constC, hC1, hC2, T)
return sum(loss.*T)
end

"""
    ``gwgrad(constC, hC1, hC2, T)``
Fast calculation of the cost matrix for an iteration of the conditional gradient descent for the calculation of the Gromov–Wasserstein distance
(see [`condgraddesc`](@ref)). The gradient is computed as described in Proposition 2 of Peyré et al. (2016).

# References

G. Peyré, M. Cuturi, and J. Solomon, "Gromov-Wasserstein averaging of kernel and distance matrices."
International Conference on Machine Learning (ICML). 2016.
"""
function gwgrad(constC, hC1, hC2, T)
    return 2*tensor_product(constC, hC1, hC2, T)
end


"""
    ``solve_linesearch(coupl, delta_coupl, grad_cost, loss, coupl_up; C1, C2, constC)``

Finding the optimal step size for the conditional gradient descent for the calculation of the Gromov–Wasserstein distance (see [`condgraddesc`](@ref)).
The algorithm is a verion of Algorithm 2 proposed in Vayer et al. (2019).

# Arguments
- `coupl`: An ``m x n`` matrix that ecodes the optimal coupling at a given iteration of the gradient descent.
- `delta_coupl`: An ``m x n`` matrix that describes the difference between the optimal coupling from the previous iteration and the one found in the current gradient step.
- `grad_cost`:  The ``m x n`` cost matrix of the linearized transport problem. Corresponds to the gradient of the cost.
- `loss`: Value of the cost at `coupl`.
- `coupl_up`: An ``m x n`` matrix that corresponds to the optimal coupling found by linearization in the current gradient step.
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `constC`: Matrix of dimension ``m x n`` required for the fast computation of the Gromov-Wasserstein gradient (see Peyré et al. (2016)).

# Returns
- `alpha` : The optimal step size of the conditional gradient descent.

# References

G. Peyré, M. Cuturi, and J. Solomon, "Gromov-Wasserstein averaging of kernel and distance matrices."
International Conference on Machine Learning (ICML). 2016.

T. Vayer, L. Chapel, R. Flamary, R. Tavenard and N. Courty.
"Optimal Transport for structured data with application on graphs"
International Conference on Machine Learning (ICML). 2019.
"""
function solve_linesearch(coupl, delta_coupl, grad_cost, loss, coupl_up; C1, C2, constC)


    tmp = C1*delta_coupl*C2
    a = -2 * sum(tmp .* delta_coupl)
    b = sum(constC .* delta_coupl) - 2 *(sum(tmp .* coupl) + sum(C1*coupl*C2 .* delta_coupl))
    
        

    if a > 0  # convex
        alpha = min(1, max(0, -b/ (2.0 * a)))
    else  # non convex
        if a+b < 0
             alpha = 1
        else
            alpha = 0
        end
    end

    return alpha
end

"""
    ``condgraddesc(a, b, coupl;  C1, C2, constC, numItermax = 200, numItermaxEmd = 10_000_000, stopThr = 1e-9, stopThr2 = 1e-9, verbose = false, log = false, cuda = false, safe = false)``

Implements a conditional gradient descent for the calculation of the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)). The algorithm is a verion of the one
implemented by Flamary et al. (2021).

# Arguments
- `a`: Probability vector of length ``m``.
- `b`: Probability vector of length ``n``.
- `coupl`: An ``m x n`` matrix that encodes a coupling between ``a`` and ``b``.
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `constC`: Matrix of dimension ``m x n`` required for the fast computation of the Gromov-Wasserstein gradient (see Peyré et al. 2016).
- `numItermax`: Maximal number of iterations for the conditional gradient descent.
- `numItermaxEmd`: Maximal number of iterations for solving the optimal transport probblem in each gradient step.
- `stopThr1`:  Stop threshold on the relative variation (>0).
- `stopThr2`:  Stop threshold on the absolute variation (>0).
- `log`: Boolean that signals if the progress of the gradient descent should be documented and returned (a log will be returned, if ``log == true``).
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `safe`: Boolean that signals wheter it is ensured that the cost in the linear problem of one gradient step are positive (this will be ensured, if ``safe == true``).

# Returns
- `res` : The value of the Third Lower Bound.
- `coupl` : Optimal coupling between ``(C1, p)`` and ``(C2, q)``.
- `log` : A vector that documents the progess of the gradient descent (only returned if ``log == true``) .

# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

G. Peyré, M. Cuturi, and J. Solomon, "Gromov-Wasserstein averaging of kernel and distance matrices."
International Conference on Machine Learning (ICML). 2016.

S. Chowdhury and F. Mémoli."The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.

R. Flamary, N. Courty, A. Gramfort, M.. Alaya, A. Boisbunon, S. Chambon, L. Chapel,
A. Corenflos, K. Fatras, N. Fournier, L. Gautheron, N. Gayraud, H. Janati, A. Rakotomamonjy,
I. Redko, A. Rolet, A. Schutz, V. Seguy, D. Sutherland, R. Tavenard, A. Tong and T. Vayer.
"POT Python Optimal Transport library".
Journal of Machine Learning Research, 22(78):1−8, 2021.
"""
function condgraddesc(a, b, coupl;  C1, C2, constC, numItermax = 200, numItermaxEmd = 10_000_000, stopThr = 1e-9, stopThr2 = 1e-9, verbose = false, log = false, cuda = false, safe = false)

    a = Array{Float64}(a)
    a ./=sum(a)
    b = Array{Float64}(b)
    b ./=sum(b)

    if log
        log_vec = zeros(1)
    end

 

    loss = gw_loss(constC, C1, 2*C2, coupl)

    if log
        log_vec[1] = loss
    end

    if verbose
        print(" It = 1 \n Loss = $loss \n Relative Loss = 0 \n Absolute Loss = 0")
        now()
        print("\n")

    end

    for it in 1:numItermax

        old_loss = loss

        # problem linearization
        grad_cost = gwgrad(constC, C1, 2*C2, coupl)

        # set grad_cost positive
        if safe
            grad_cost .+= abs(minimum(grad_cost))
        end

        # solve linear program
        grad_cost = Array{Float64}(grad_cost)
        _, coupl_up = emd_bonneel_with_plan(a, b, grad_cost, max_iter = numItermaxEmd)

        if cuda
            coupl_up = cu(coupl_up)
        end
        
        delta_coupl = coupl_up - coupl

        # line search
        alpha = solve_linesearch(coupl, delta_coupl, grad_cost, loss, coupl_up, C1 = C1, C2 = C2, constC = constC)

        coupl = coupl + alpha * delta_coupl
        
        loss = gw_loss(constC, C1, 2*C2, coupl)

        # test convergence
        abs_delta_loss = abs(loss - old_loss)
        relative_delta_loss = abs_delta_loss / abs(loss)

        if (relative_delta_loss < stopThr) || (abs_delta_loss < stopThr2)
            break
        end

        if log
            append!(log_vec, loss)
        end
                
        if verbose
            if mod(it, 10) == 0
                print(" It = $it \n Loss = $loss \n Relative Loss = $relative_delta_loss \n Absolute Loss = $abs_delta_loss")
                now()
                print("\n")
            end
        end
    end

    if log
        return (coupl, log_vec)
    else
        coupl 
    end
end

"""
    ``gromov_wasserstein(C1, C2, p, q; log = false, cuda = false, st_coupl = nothing, verbose = false, kwargs...)``

Returns the Gromov–Wasserstein distance (see Mémoli (2011) and Chowdhury and Mémoli (2019)) between ``(C1, p)`` and ``(C2, q)``. The algorithm is a version
of the one proposed by Flamary et al. (2021).

# Arguments
- `C1`: Symmetric cost matrix of dimension ``m x m``.
- `C2`: Symmetric cost matrix of dimension ``n x n``.
- `p`: Probability vector of length ``m``.
- `q`: Probability vector of length ``n``.
- `log`: Boolean that signals if the progress of the gradient descent should be documented and returned (a log will be returned, if ``log == true``).
- `cuda`: Boolean that signals the usage of GPU (if ``true`` GPU is used).
- `st_coupl`: An ``m x n`` matrix that encodes a coupling of `p` and `q` and acts as starting point of the gradient descent (optional).
- `verbose`: Boolean that signals if progress should be displayed at `stdout` (progress will be displayed, if ``verbose == true``).
- `kwargs`: Additional arguments to pass to [`condgraddesc`](@ref) such as `numItermax` and `numItermaxEmd`.

# Returns
- `res` : The value of the Gromov-Wasserstein distance.
- `coupl` : Optimal coupling between ``(C1, p)`` and ``(C2, q)``.
- `log` : A vector that documents the progess of the gradient descent (only returned if ``log == true``) .


# References

F. Mémoli. "Gromov–Wasserstein distances and the metric approach to object matching". Foundations of computational
mathematics 11.4: 417-487, 2011.

S. Chowdhury and F. Mémoli."The Gromov-Wasserstein distance between networks and stable network invariants".
Information and Inference: A Journal of the IMA, 8(4), 757-787, 2019.

R. Flamary, N. Courty, A. Gramfort, M.. Alaya, A. Boisbunon, S. Chambon, L. Chapel,
A. Corenflos, K. Fatras, N. Fournier, L. Gautheron, N. Gayraud, H. Janati, A. Rakotomamonjy,
I. Redko, A. Rolet, A. Schutz, V. Seguy, D. Sutherland, R. Tavenard, A. Tong and T. Vayer.
"POT Python Optimal Transport library".
Journal of Machine Learning Research, 22(78):1−8, 2021.

"""
function gromov_wasserstein(C1, C2, p, q; log = false, cuda = false, st_coupl = nothing, verbose = false, kwargs...)



    if isnothing(st_coupl)
        st_coupl = p*q'
    else
        # Check marginals of st_coupl
        t1 = all(x -> isapprox(x..., atol = 1e-8), zip(sum(st_coupl, dims = 1), q))
        t2 = all(x -> isapprox(x..., atol = 1e-8), zip(sum(st_coupl, dims = 2), p))
        if !(t1 && t2)
            error("Invalid start coupling")
        end
    end

    if cuda
        C1 = cu(C1)
        C2 = cu(C2)
        st_coupl = cu(st_coupl)
        p = cu(p)
        p ./=sum(p)
        q = cu(q)
        q ./=sum(q)
    end

    if cuda
        constC = C1.^2*p*CUDA.ones(1,length(q)) + CUDA.ones(length(p))*q'C2.^2
    else
        constC = C1.^2*p*ones(1,length(q)) + ones(length(p))*q'C2.^2
    end

    if log
        coupl, log_vec = condgraddesc(p, q,  st_coupl; C1 = C1, C2 = C2, constC = constC, log = true, cuda = cuda, verbose = verbose, kwargs...)
        res = gw_loss(constC, C1, 2*C2, coupl)
        result = (res, coupl, log_vec)
    else
        coupl = condgraddesc(p, q,  st_coupl; C1 = C1, C2 = C2, constC = constC, log = false, cuda = cuda, verbose = verbose, kwargs...)
        res = gw_loss(constC, C1, 2*C2, coupl)
    result = (res, coupl)
end

return result

end
