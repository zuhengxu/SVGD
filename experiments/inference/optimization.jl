using Parameters

# OptimizationAlgorithm type
abstract type OptAlg end

@with_kw struct GD <: OptAlg
    lrt::Function
    niters::Int64
    past0::Function = x-> zero(x)
end

@with_kw struct Momentum <: OptAlg
    lrt::Function #constant
    niters::Int64
    γ::Float64 = 0.9
    past0::Function= x-> zero(x)
end

@with_kw struct AdaGrad <: OptAlg
    lrt::Function = i-> 0.01
    niters::Int64 
    ϵ::Float64 = 1e-8
    past0::Function = x-> zero(x)
end

# RMSprop = Adagrad + Momentum
@with_kw struct RMSprop <: OptAlg
    lrt::Function = i-> 0.001#constant
    niters::Int64
    γ::Float64 = 0.9
    ϵ::Float64  = 1e-8
    past0::Function= x-> zero(x)
end

@with_kw struct WAG <: OptAlg
    lrt::Function
    niters::Int64
    α::Float64 # > 3
    past0::Function= x-> x
end    
    
@with_kw struct WNes <: OptAlg
    lrt::Function #constant
    niters::Int64
    u::Float64  # upper bounds on the Lip const of ∇log
    b::Float64  # shrinkage parameter > 0
    past0::Function= x-> x
end



# Update rule for optimization (be careful on the sign of ∇)
function STEP!(alg::GD, k::Int64, θ::Array{Float64, 2}, ∇::Array{Float64, 2}, past::Array{Float64, 2})
    θ .+= alg.lrt(k)*∇ 
end


# note that the past start from zero(x0)
function STEP!(alg::Momentum, k::Int64, θ::Array{Float64, 2},  ∇::Array{Float64,2}, past::Array{Float64,2})
    v = alg.γ * past .- alg.lrt(k) * ∇
    θ .-= v
    past .= v
end

function STEP!(alg::AdaGrad, k::Int64, θ::Array{Float64, 2}, ∇::Array{Float64, 2}, past::Array{Float64, 2})
    eps = alg.lrt(k)
    past .+= ∇.^2
    θ .+=  (eps ./ sqrt.(past .+ eps)) .* ∇
end

function STEP!(alg::RMSprop, k::Int64, θ::Array{Float64, 2}, ∇::Array{Float64, 2}, past::Array{Float64, 2})
    past .= alg.γ * past + (1.0- alg.γ) * ∇.^2
    θ .+=  (alg.lrt(k) ./ sqrt.(past .+ alg.ϵ)) .* ∇
end

#past start from x0
function STEP!(alg::WAG, k::Int64, θ::Array{Float64, 2}, ∇::Array{Float64, 2}, past::Array{Float64, 2})
    adjust = (k-1.)/k *(θ .- past) .+ (k + alg.α -2. )/k *alg.lrt(k)*∇
    past .= θ .+ alg.lrt(k)*∇ 
    θ .= past .+ adjust
end

function STEP!(alg::WNes, k::Int64, θ::Array{Float64, 2}, ∇::Array{Float64, 2}, past::Array{Float64, 2})
    eps = alg.lrt(k)
    b = alg.b
    u = alg.u
    C = 1. + b - 2. *(1. +b)*(2. + b)*u*eps/(sqrt(b^2 + 4. *(1. +b)*u*eps) - b + 2. *(1. + b)*u*eps)
    adjust = C*past
    past .= θ .+ alg.lrt(k)*∇ 
    θ .= (1. +C)*past .- adjust
end


