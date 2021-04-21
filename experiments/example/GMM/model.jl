############
#We parameterize it such that the posterior has full support
# 1. λ1....λk ∼ LogGamma(1, 1) (then θ = softmax(λ) ∼ Dir(1))
# 2. τ_kd ∼ LogNormal(0,1) (σ = exp(τ) ∼ N(0,1) )
# 3. μ1...μk ∼ N(0,I)
# 4. x|λ, μ, σ ∼  ∑_k θ_k N(μ_k, diag(σ_k1, .., σ_kD) )
############


############
#log posterior
############
function logπ(Z, X, K, D)
# Z = ( μ1...μK, λ1....λK,τ_11..τ_1D....τ_KD)
# K = number of cluster, D = dim of the data
# X ∈ R_DN
    N = size(X,2)
    Mu = @view Z[1:K*D]
    Lambda = @view Z[K*D + 1: K*D+ K]
    Tau = @view Z[K*D+ K+1 : end]
    MM = reshape(Mu, (D,K))
    X = reshape(X, (D,1,N))

    logprior = 0.5* sum(Mu.^2) + sum(Lambda .- expm1.(Lambda) .- 1.)-0.5*sum(Tau.^2)
    term1= Lambda .- logsumexp(Lambda) - 0.5*dropdims(sum(reshape(Tau, (D, K)).^2, dims = 1) , dims = 1)
    term2 = -0.5*dropdims(sum((X .- MM).^2 .*(expm1.(-reshape(Tau, (D, K,1)).^2) .+ 1.),dims = 1), dims = 1)

    llh = sum(mapslices(logsumexp, term1 .+ term2, dims = 1))
    return logprior .+ llh
end

# X = dat["X"]
# Z = ones(15)
# K = 3
# D = 2
# N = 400
# Xs = reshape(X, (D,1,N))
# Mu = Z[1:K*D]
# Lambda = Z[K*D + 1: K*D+ K]
# Tau = Z[K*D+ K+1 : end]
# MM = reshape(Mu, (D,K))



##########
# compute lpdf/∇lpdf/Hessian with dataset
##########
dat = load("example/GMM/data/dataset.jld")
lpdf = θ-> logπ(θ, dat["X"], 3,2)
∇lpdf = θ -> ∇logπ(θ, lpdf)
Hessian = θ -> ForwardDiff.hessian(lpdf, θ)

