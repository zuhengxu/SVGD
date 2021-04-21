#########################################################################################
# Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
# The observed data D = {X, y} consist of N binary class labels, 
# y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
# The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
# and a precision parameter \alpha \in R_+. We assume the following model:
#     p(α) = Gamma(α ; a, b) , τ = log α ∈ R  
#     p(w_k | τ) = N(w_k; 0, exp(-τ))
#     p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t)), y ∈ {1, 0}
#########################################################################################



##########
# prior sampler (used for init particles)
##########
function prior_sampler(N, d;a = 1., b = 0.01)
    as = rand(Gamma(a, 1/b), N)
    τs = log1p.(as .- 1.)
    W = randn(N, d) .* sqrt.(1 ./as)
    return Matrix([τs W]')
end

# θ0= prior_sampler(100, 3)



##########
# log posterior
##########
function logπ(θ, Y, X; a =1., b = 0.01)
#θ ∈ Rd, X ∈ R_{N(d-1)}, Y ∈ RN 
    d = size(X,2)
    τ = θ[1]
    W = θ[2:end]

    logpτ = a*τ - b*(expm1(τ) +1.)
    logpW = 0.5*(d*τ - (expm1(τ) +1.)* sum(W.^2))
    llh = sum((Y .-1.) .* (X*W) .- log1p.(expm1.(-X*W)  .+1.) )

    return logpτ + logpW +llh
end


##########
# compute lpdf/∇lpdf/Hessian with dataset
##########
dat = load("example/logistic_reg/data/dataset.jld")
lpdf = θ-> logπ(θ, dat["Y"], dat["X"])
∇lpdf = θ -> ∇logπ(θ, lpdf)
Hessian = θ -> ForwardDiff.hessian(lpdf, θ)










