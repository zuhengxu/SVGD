using Distances, LinearAlgebra, Random, Distributions, Suppressor, Parameters
include("kernel.jl")
include("optimization.jl")
include("anneal.jl")



# SVGD struct 
@with_kw struct SVGD
	init_ptc::Array{Float64,2} 
    lpdf::Function
    ∇lpdf::Function
    kernel::Function
    anneal::Function = id_sched
end


# SVGD_bw struct 
@with_kw struct SVGD_bw
    init_ptc::Array{Float64,2} 
    lpdf::Function
    ∇lpdf::Function
    Hessian::Function
    bw_rule::Function = curvature_scaling # function that compute local bw for each ptc
    kernel::Function = RBFkernel_bw! # must choose RBFkernel_curv
    anneal::Function = id_sched
end


# SVGD_Gauss struct 
@with_kw struct SVGD_Gauss
    init_ptc::Array{Float64,2} 
    lpdf::Function
    ∇lpdf::Function
    Hessian::Function
    Σ_rule::Function = Hessian_diag # function that compute local bw for each ptc
    kernel::Function = Gausskernel! # must choose RBFkernel_curv
    anneal::Function = id_sched
end




##########################
# SVGD trace/sampler(no trace save)
##########################
function svgd_trace(self::SVGD, alg::OptAlg, ntrace::Int64, bw::Float64)
    ############
    # bw > 0: use given bw 
    # bw < 0: use meian trick-->  bw = med^2/log(nptc), med is the median of the pairwise L2distance of current ptcs.
    ############

    #initial particles
    θ =	copy(self.init_ptc) #using copy then it wont overwrite the struct
    d, n = size(θ)
    #save the ptc every trace_each iters
    trace_each = ceil(alg.niters/ntrace)
    trace = zeros(d, n, ntrace+1)
    count = 1
    trace[:,:,count] = θ
    
    past = alg.past0(θ)
    K = zeros(n,n)
    ∇K = similar(θ)
    for i in 1:alg.niters
        # compute kernel matrix and ∇K 
        self.kernel(K, ∇K, θ, bw)
        # compute functional grad
        grad = 1. /n * (self.anneal(i)*self.∇lpdf(θ)* K + ∇K)
        # perform a step
        STEP!(alg, i, θ, grad, past)
        if i % trace_each == 0
            count += 1
            trace[:,:, count] .= θ   
            print(count, "/",ntrace+1, "\r")
            flush(stdout)
        end
    end
    return trace
end

function svgd_sampler(self::SVGD, alg::OptAlg, bw::Float64)
    ############
    #bw >0 : use given bw 
    #bw <0 : use meian trick-->  bw = med^2/log(nptc), med is the median of the pairwise distance of current ptcs.
    ############
    
    #initial particles
    θ =	copy(self.init_ptc) #using copy then it wont overwrite the struct
    d, n = size(θ)

    past = alg.past0(θ)
    K = zeros(n,n)
    ∇K = similar(θ)
    for i in 1:alg.niters
        # compute kernel matrix and ∇K 
        self.kernel(K, ∇K, θ, bw)
        # compute functional grad
        grad = 1. /n * (self.anneal(i)*self.∇lpdf(θ)* K + ∇K)
        # perform a step
        STEP!(alg, i, θ, grad, past)
    
        if i % 100 == 0
            print(i, "/",alg.niters, "\r")
            flush(stdout)
        end
    end
    return θ
end





#####################
# SVGD(bw using curvature scaling) trace/sampler 
#####################
function svgd_trace(self::SVGD_bw, alg::OptAlg, ntrace::Int64)

    #initial particles
    θ =	copy(self.init_ptc) #using copy then it wont overwrite the struct
    d, n = size(θ)
    #save the ptc every trace_each iters
    trace_each = ceil(alg.niters/ntrace)
    trace = zeros(d, n, ntrace+1)
    count = 1
    trace[:,:,count] = θ

    past = alg.past0(θ)
    K = zeros(n,n)
    ∇K = similar(θ)

    for i in 1:alg.niters
        # compute kernel matrix and ∇K 
        self.kernel(K, ∇K, θ, self.Hessian, self.bw_rule)
        # compute functional grad
        grad = 1. /n * (self.anneal(i)*self.∇lpdf(θ)* K + ∇K)
        # perform a step
        STEP!(alg, i, θ, grad, past)
    
        if i % trace_each == 0
            count += 1
            trace[:,:, count] .= θ   
            print(count, "/",ntrace+1, "\r")
            flush(stdout)
        end
    end
    return trace
end

function svgd_sampler(self::SVGD_bw, alg::OptAlg)

    #initial particles
    θ =	copy(self.init_ptc) #using copy then it wont overwrite the struct
    d, n = size(θ)

    past = alg.past0(θ)
    K = zeros(n,n)
    ∇K = similar(θ)

    for i in 1:alg.niters
        # compute kernel matrix and ∇K 
        self.kernel(K, ∇K, θ, self.Hessian, self.bw_rule)
        # compute functional grad
        grad = 1. /n * (self.anneal(i)*self.∇lpdf(θ)* K + ∇K)
        # perform a step
        STEP!(alg, i, θ, grad, past)
    
        if i % 100 == 0
            print(i, "/",alg.niters, "\r")
            flush(stdout)
        end
    end
    return θ
end




#####################
# SVGD(using general gaussian kernel) trace/sampler 
#####################
function svgd_trace(self::SVGD_Gauss, alg::OptAlg, ntrace::Int64)

    #initial particles
    θ =	copy(self.init_ptc) #using copy then it wont overwrite the struct
    d, n = size(θ)
    #save the ptc every trace_each iters
    trace_each = ceil(alg.niters/ntrace)
    trace = zeros(d, n, ntrace+1)
    count = 1
    trace[:,:,count] = θ

    past = alg.past0(θ)
    K = zeros(n,n)
    ∇K = similar(θ)

    for i in 1:alg.niters
        # compute kernel matrix and ∇K 
        self.kernel(K, ∇K, θ, self.Hessian, self.Σ_rule)
        # compute functional grad
        grad = 1. /n * (self.anneal(i)*self.∇lpdf(θ)* K + ∇K)
        # perform a step
        STEP!(alg, i, θ, grad, past)
    
        if i % trace_each == 0
            count += 1
            trace[:,:, count] .= θ   
            print(count, "/",ntrace+1, "\r")
            flush(stdout)
        end
    end
    return trace
end

function svgd_sampler(self::SVGD_Gauss, alg::OptAlg)

    #initial particles
    θ =	copy(self.init_ptc) #using copy then it wont overwrite the struct
    d, n = size(θ)

    past = alg.past0(θ)
    K = zeros(n,n)
    ∇K = similar(θ)

    for i in 1:alg.niters
        # compute kernel matrix and ∇K 
        self.kernel(K, ∇K, θ, self.Hessian, self.Σ_rule)
        # compute functional grad
        grad = 1. /n * (self.anneal(i)*self.∇lpdf(θ)* K + ∇K)
        # perform a step
        STEP!(alg, i, θ, grad, past)
    
        if i % 100 == 0
            print(i, "/",alg.niters, "\r")
            flush(stdout)
        end
    end
    return θ
end


