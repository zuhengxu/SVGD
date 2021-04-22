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
niters = 2000
rms = RMSprop(lrt = i-> .01, niters = niters)
#result_dir
result_dir  = "example/GMM/result"

##########
# build svgd(median bw) and svgd(localized kernel)
########## 
svgd = SVGD(init_ptc = x0, lpdf = lpdf, ∇lpdf = ∇lpdf, kernel = RBFkernel!)
asvgd = SVGD(init_ptc = x0, lpdf = lpdf, ∇lpdf = ∇lpdf, kernel = RBFkernel!, anneal = i->  cyclical_sched(i, 1000, 1.))
svgd_local = SVGD_Gauss(init_ptc = x0, lpdf = lpdf, ∇lpdf= ∇lpdf, Hessian = Hessian)
asvgd_local = SVGD_Gauss(init_ptc = x0, lpdf = lpdf, ∇lpdf= ∇lpdf, Hessian = Hessian, anneal = i->  cyclical_sched(i, 1000, 1.))


##########
# Getting traces
########## 
ntrace = 200
Tsv = svgd_trace(svgd, rms, ntrace, -1.)
Tasv = svgd_trace(asvgd, rms, ntrace, -1.)
Tsv_local  = svgd_trace(svgd_local, rms, ntrace)
Tasv_local = svgd_trace(asvgd_local, rms, ntrace)
#save the traces
save(joinpath(result_dir, "Traces.jld"), "sv", Tsv, "asv", Tasv,
                                        "sv_local", Tsv_local, 
                                        "acv_local", Tasv_local)


##########
# quantitative comparison
########## 
# ksd over all iterations 
ksd_trace(Tsv, x-> ForwardDiff.gradient(lpdf, x))
ksd_trace(Tsv_local, x-> ForwardDiff.gradient(lpdf, x))
ksd_trace(Tasv, x-> ForwardDiff.gradient(lpdf, x))
ksd_trace(Tasv_local, x-> ForwardDiff.gradient(lpdf, x))



# ELBO estimation Eq[log p(x, Z)] 
elbo_trace(Tasv_local, lpdf)
elbo_trace(Tasv, lpdf)
elbo_trace(Tsv, lpdf)
elbo_trace(Tsv_local,lpdf)