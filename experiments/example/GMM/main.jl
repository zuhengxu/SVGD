#pwd() = "qua5/experiemnts"
include("inference/SVGD.jl")
include("example/common/result.jl")
include("example/common/util.jl")
include("example/common/plotting.jl")
include("example/GMM/model.jl")


##########
# set up 
##########
Random.seed!(2021);
# init paritcles from prior
x0 = prior_sampler(200, 2,2)
# #RMSprop update rule
niters = 200
rms = RMSprop(lrt = i-> .05, niters = niters)
#result_dir
result_dir  = "example/logistic_reg/result"

##########
# build svgd(median bw) and svgd(localized kernel)
########## 
svgd = SVGD(init_ptc = x0, lpdf = lpdf, ∇lpdf = ∇lpdf, kernel = RBFkernel!)
svgd_local = SVGD_Gauss(init_ptc = x0, lpdf = lpdf, ∇lpdf= ∇lpdf, Hessian = Hessian)

##########
# Getting traces
########## 
Tsv = svgd_trace(svgd, rms, 200, -1.)
Tsv_local  =svgd_trace(svgd_local, rms, 200)
#save the traces
save(joinpath(result_dir, "Traces.jld"), "sv", Tsv, "sv_local", Tsv_local)


##########
# quantitative comparison
########## 
# ksd over all iterations






# logposterior value
