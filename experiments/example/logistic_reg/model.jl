#########################################################################################
# Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
# The observed data D = {X, y} consist of N binary class labels, 
# y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
# The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
# and a precision parameter \alpha \in R_+. We assume the following model:
#     p(α) = Gamma(α ; a, b) , τ = log α ∈ R  
#     p(w_k | a) = N(w_k; 0, exp(-τ)^-1)
#     p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t)), y ∈ {+1, -1}
#########################################################################################


function logπ(θ, dat; batchsize = 100, a =1., b = 0.01)
    d = size(dat,2) - 1
    τ = θ[1]
    W = θ[2:end]
    Y = dat[:,1]
    X = dat[:,2:end]

    logpτ = a*τ - b*(expm1(τ) +1.)
    logpW = 0.5*(d*τ - (expm1(τ) +1.)* sum(W.^2))
    llh = sum(-0.5*(Y .+1.) .* (X*w) .- log1p.(expm1.(-X*W)  .+1.) )

    return logpτ + logpW +llh
end





##########
# prior sampler (used as init particles)
##########

function prior_sampler(Z)
    return randn(100, 2)
end
