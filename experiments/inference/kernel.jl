using Distances, ForwardDiff, LinearAlgebra, Random, Distributions, Suppressor, Statistics
using Base.Threads: @threads

#################
## RBFkernel 
#################
function RBFkernel(θ::Array{Float64, 2}, bw::Float64) 
    n = size(θ, 2)
    #compute pairwise L2^2 distance
    K_euc = pairwise(SqEuclidean(), θ)

    #if bw<0 : use median trick
    bw = bw < 0. ? median(K_euc)/log(n) : bw
    #compute kernel matrix
    KM = expm1.((-0.5/bw) *K_euc) .+ 1.
    ∇K = similar(θ)
    D = similar(θ)
    #computing the KM and ∇K with multithread
    @views @inbounds @threads for i in 1:n
        D .= θ .- θ[:, i]
        #∇K[:, j] = ∑_i ∇ K(Xi, Yj)
        ∇K[:, i] = -(1. / bw)*D * KM[:, i]
    end
    return KM , ∇K
end
# RBFkernel! (bw <0: use median trick; bw >0: use given value)
function RBFkernel!( KM::Array{Float64, 2}, ∇K::Array{Float64, 2}, θ::Array{Float64, 2}, bw::Float64) 
    n = size(θ, 2)
    K_euc = pairwise(SqEuclidean(), θ)
    #if bw<0 : use median trick
    bw = bw < 0. ? median(K_euc)/log(n) : bw
    #compute kernel matrix
    KM .= expm1.((-0.5/bw) *K_euc) .+ 1.
    D = similar(θ)
    #computing the KM and ∇K with multithread
    @views @inbounds @threads for i in 1:n
        D .= θ .- θ[:, i]
        #∇K[:, j] = ∑_i ∇ K(Xi, Yj)
        ∇K[:, i] = -(1. / bw)*D * KM[:, i]
    end
    return KM , ∇K
end


# RBFkernel using local bw (bw = curvature)
function RBFkernel_bw!( KM::Array{Float64, 2}, ∇K::Array{Float64, 2}, θ::Array{Float64, 2},H::Function, bw_rule::Function) 
    n = size(θ, 2)
    # compute pairwise square euc distance
    K_euc = pairwise(SqEuclidean(), θ)
    D = similar(θ)
    #computing the KM and ∇K with multithread
    @views @inbounds @threads for i in 1:n
        #bw for each particle
        bw = bw_rule(θ[:, i], H)
        #local kernel matrix 
        KM[:, i] = expm1.((-0.5/bw) *K_euc[:, i]) .+ 1.
        #∇K[:, j] = ∑_i ∇ K(Xi, Yj)
        D .= θ .- θ[:, i]
        ∇K[:, i] = -(1. / bw)*D * KM[:, i]
    end
    return KM , ∇K
end

# bw = inverse of the maixmal diag of ∇^2 lpdf(x_i) 
function curvature_scaling(x, H::Function)
    return 1. /(maximum(abs.(Diagonal(H(x)))) + 1e-10)
end

#################
## anisoptropical gaussian kernel 
#################

function Gausskernel!(KM::Array{Float64, 2}, ∇K::Array{Float64, 2}, θ::Array{Float64, 2},H::Function, Σ_rule::Function)
    n = size(θ, 2)
    D = similar(θ)
    K = zeros(n,n)
    #computing the KM and ∇K with multithread
    @views @inbounds @threads for i in 1:n
        #bw for each particle
        Q =  Σ_rule(θ[:, i], H)
        # compute pairwise square Mahalanobis distance
        K .= pairwise(SqMahalanobis(Q), θ)
        #local kernel matrix 
        KM[:, i] = expm1.(-0.5*K[:, i]) .+ 1.
        #∇K[:, j] = ∑_i ∇ K(Xi, Yj)
        D .= θ .- θ[:, i]
        ∇K[:, i] = -Q*D * KM[:, i]
    end
    return KM , ∇K
end


# Σ = inv diag hessian
function Hessian_diag(x, H::Function)
    return  Matrix(Diagonal(abs.(H(x)) .+ 1e-10))
end

# Q = Hessian_inv(1., x -> 3*ones(3,3))
# θ = randn(3,10)
# pairwise(SqMahalanobis(Q), θ)
# pairwise(SqEuclidean(), θ)