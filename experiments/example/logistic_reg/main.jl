#pwd() = "qua5/experiemnts"
include("inference/SVGD.jl")
include("example/common/result.jl")
include("example/common/util.jl")
include("example/common/plotting.jl")
include("example/logistic_reg/model.jl")

##########
# set up 
##########
Random.seed!(2021);
# init paritcles from prior
x0 = prior_sampler(200, 3)
# #RMSprop update rule
niters = 3000
rms = RMSprop(lrt = i-> .01, niters = niters)
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
Tsv = svgd_trace(svgd, rms, 300, -1.)
Tsv_local  =svgd_trace(svgd_local, rms, 300)
#save the traces
save(joinpath(result_dir, "Traces.jld"), "sv", Tsv, "sv_local", Tsv_local)


##########
# quantitative comparison
########## 
# ksd over all iterations 
ksv = ksd_trace(Tsv, x-> ForwardDiff.gradient(lpdf, x));
ksv_local = ksd_trace(Tsv_local, x-> ForwardDiff.gradient(lpdf, x));


# ELBO estimation Eq[log p(x, Z)] 
esv = elbo_trace(Tsv, lpdf);
esv_local = elbo_trace(Tsv_local,lpdf);


#save the ELBO and KSD
save(joinpath(result_dir, "ksd.jld"), "sv", ksv, "sv_local", ksv_local);

save(joinpath(result_dir, "elbo.jld"), "sv", esv, "sv_local", esv_local)
