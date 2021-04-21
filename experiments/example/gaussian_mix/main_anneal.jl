#pwd() = "qua5/experiemnts"
include("inference/SVGD.jl")
include("example/common/result.jl")
include("example/common/util.jl")
include("example/common/plotting.jl")
include("example/gaussian_mix/model.jl")

##########
# set up 
##########
Random.seed!(2021);
# two different init ptcs(N = 500) mu1 = (10, 10), mu2 = (0,0) 
x0  = rand(Normal(0,0.5), (2,500));
x10 = rand(Normal(10, 0.5), (2, 500));
# number of iterations
niters = 1000
#RMSprop update rule
rms = RMSprop(lrt = i-> .05, niters = niters)

#plotting params
x = -7:.1:7
y = -7:.1:7
f = (x, y) -> exp(lpdf([x, y]))
#figure path
fig_path = "example/gaussian_mix/figure/"

##########
##########
# 1. get SVGD/ASVGD trace (RMSprop) under 2 different init ptc 
##########
##########
# set up SVGD and A-SVGD
SV0 = SVGD(x0, lpdf, ∇lpdf, RBFkernel!, id_sched)
SV0_anneal = SVGD(x0, lpdf, ∇lpdf, RBFkernel!, i->  cyclical_sched(i, 500, 1.))
SV10 = SVGD(x10, lpdf, ∇lpdf, RBFkernel!, id_sched)
SV10_anneal = SVGD(x10, lpdf, ∇lpdf, RBFkernel!, i->  cyclical_sched(i, 500, 1.))

# get the trace
T0_rms = svgd_trace(SV0, rms, 100, .5);
T0_rms_anneal = svgd_trace(SV0_anneal, rms, 100,.5);
T10_rms = svgd_trace(SV10, rms, 100, .5);
T10_rms_anneal = svgd_trace(SV10_anneal, rms, 100,.5);
# save traces
save("example/gaussian_mix/result/result1.jld", "T0", T0_rms, "T10", T10_rms, "T0_anneal", T0_rms_anneal, "T10_anneal", T10_rms_anneal)

######
# plots
######
# contour plot save
contour_save(x, y, f, T0_rms[:,:,end]; folder = fig_path, name = "T0_rms.png", title = "no anneling, mu0 = (0,0)");
contour_save(x, y, f, T0_rms_anneal[:,:,end]; folder = fig_path, name = "T0_rms_anneal.png", title = "with anneling, mu0 = (0,0)");
contour_save(x, y, f, T10_rms[:,:,end]; folder = fig_path, name = "T10_rms.png",title = "no anneling, mu0 = (10,10)");
contour_save(x, y, f, T10_rms_anneal[:,:,end]; folder = fig_path, name = "T10_rms_anneal.png",title = "with anneling, mu0 = (10,10)");
# # gif save
gif2d(x, y, f, T0_rms; folder = fig_path, name = "T0_rms.gif");
gif2d(x, y, f, T0_rms_anneal; folder = fig_path, name = "T0_rms_anneal.gif");
gif2d(x, y, f, T10_rms; folder = fig_path, name = "T10_rms.gif");
gif2d(x, y, f, T10_rms_anneal; folder = fig_path, name = "T10_rms_anneal.gif");


# moment estimates (niters vs Ex2) and (niters vs Ex)







